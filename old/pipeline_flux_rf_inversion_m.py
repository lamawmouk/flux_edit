# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# import wandb
# wandb.init(project="rf-inversion-debug", name="inversion-run-1")

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

import torchvision.utils as vutils
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FluxLoraLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name



# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    """This function computes a shift value (mu) based on the image sequence length.
It's used to adjust how much the latent space representation shifts at each diffusion step.
The formula is a simple linear equation that interpolates between base_shift and max_shift depending on image_seq_len"""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

""""""
# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    """This function retrieves latent representations of an image from the VAE encoder.
If the encoder has a latent_dist property, it samples from it.
Otherwise, it looks for latents directly.
If neither exists, it raises an error."""
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class FluxRFInversionPipeline(DiffusionPipeline, FluxLoraLoaderMixin):
    r"""
    The Flux pipeline for image inpainting.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Args:
        transformer ([`FluxTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`T5TokenizerFast`):
            Second Tokenizer of class
            [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels)) if hasattr(self, "vae") and self.vae is not None else 16
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = 64
        
        self.args = args
        self.for_prompt = args.for_prompt
        self.edit_prompt = args.edit_prompt
        self.result_folder = args.result_folder
        self.image_size = args.image_size
        self.device = args.device
        self.dtype = args.dtype
        self.c_in = 3
        self.pipe = args.pipe  # Must be initialized externally
        self.image_processor = args.image_processor
        self.sam = SAM(args, log_dir=self.result_folder)


    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._get_t5_prompt_embeds
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """"This function encodes textual prompts into embeddings.
It supports both CLIP and T5 encoders.
If LoRA fine-tuning is enabled, it adjusts the LoRA scale.
"""
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]

        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._get_clip_prompt_embeds
    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        lora_scale: Optional[float] = None,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # We only use the pooled prompt output from the CLIPTextModel
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_inpaint.StableDiffusion3InpaintPipeline._encode_vae_image
    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i])
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator)

        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        return image_latents

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img.StableDiffusion3Img2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(num_inference_steps * strength, num_inference_steps)

        t_start = int(max(num_inference_steps - init_timestep, 0))
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        sigmas = self.scheduler.sigmas[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, sigmas, num_inference_steps - t_start

    def check_inputs(
        self,
        prompt,
        prompt_2,
        strength,
        height,
        width,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt_2 is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt_2`: {prompt_2} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif prompt_2 is not None and (not isinstance(prompt_2, str) and not isinstance(prompt_2, list)):
            raise ValueError(f"`prompt_2` has to be of type `str` or `list` but is {type(prompt_2)}")

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._prepare_latent_image_ids
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height // 2, width // 2, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._pack_latents
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._unpack_latents
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        height = height // vae_scale_factor
        width = width // vae_scale_factor

        latents = latents.view(batch_size, height, width, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height * 2, width * 2)

        return latents

    def prepare_latents(
        self,
        image,
        timestep,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
        gamma=1.0,
        sigmas=None,
        null_prompt_embeds=None,
        null_pooled_prompt_embeds=None,
        null_text_ids=None,
        timesteps=None,
        guidance=None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        height = 2 * (int(height) // self.vae_scale_factor)
        width = 2 * (int(width) // self.vae_scale_factor)

        shape = (batch_size, num_channels_latents, height, width)
        latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)

        if latents is not None:
            return latents.to(device=device, dtype=dtype), latent_image_ids

        image = image.to(device=device, dtype=dtype)
        image_latents = self._encode_vae_image(image=image, generator=generator)

        if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
            # expand init_latents for batch_size
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            image_latents = torch.cat([image_latents], dim=0)
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        # latents = noise
        # latents = self.scheduler.scale_noise(image_latents, timestep, noise)        
        # latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        # image_latents = self._pack_latents(image_latents, batch_size, num_channels_latents, height, width)
        # print((latents - image_latents).abs().mean())
        image_latents = self._pack_latents(image_latents, batch_size, num_channels_latents, height, width)
        ori_image_latents = image_latents.clone()
        latents = self.controlled_forward_ode(
            image_latents, 
            latent_image_ids, 
            sigmas, gamma=gamma, null_prompt_embeds=null_prompt_embeds, null_pooled_prompt_embeds=null_pooled_prompt_embeds, null_text_ids=null_text_ids, timesteps=timesteps, guidance=guidance, height=height, width=width)
        return ori_image_latents, latents, latent_image_ids
    
    def controlled_forward_ode(self, image_latents, latent_image_ids, sigmas, gamma, null_prompt_embeds, null_pooled_prompt_embeds, null_text_ids, timesteps, guidance, height, width):
        """
        Eq 8 dY_t = [u_t(Y_t) + γ(u_t(Y_t|y_1) - u_t(Y_t))]dt
        
        The function integrates the latent variables Y_t forward in time.
It introduces a controlled rectification term using a guidance parameter gamma.
The term (y_1 - Y_t) / (1 - t_i) forces the trajectory to align toward a rectified version of the diffusion process.
This correction (γ * (u_t_i_cond - u_t_i)) modifies the standard diffusion to introduce rectification flux.
        """
        device = image_latents.device
        batch_size = image_latents.shape[0]
        Y_t = image_latents.clone()
        y_1 = torch.randn_like(Y_t)
        N = len(sigmas)
        if guidance is not None:
            guidance = guidance.expand(batch_size)
        
        # print(timesteps, self.scheduler.sigmas, self.scheduler.timesteps)
        for i in range(N-1): # enumerate(timesteps):
            t_i = torch.tensor(i / (N), dtype=Y_t.dtype, device=device)
            # print(t_i)
            dt = torch.tensor(1 / (N), dtype=Y_t.dtype, device=device)
            # get the unconditional vector field

            u_t_i = self.transformer(
                hidden_states=Y_t, 
                timestep=torch.tensor(t_i, dtype=Y_t.dtype, device=device).repeat(batch_size), 
                guidance=guidance,
                pooled_projections=null_pooled_prompt_embeds,
                encoder_hidden_states=null_prompt_embeds,
                txt_ids=null_text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,                
            )[0]

            # get the conditional vector field
            u_t_i_cond = (y_1 - Y_t) / (1 - t_i)

            # controlled vector field
            # Eq 8 dY_t = [u_t(Y_t) + γ(u_t(Y_t|y_1) - u_t(Y_t))]dt
            u_hat_t_i = u_t_i + gamma * (u_t_i_cond - u_t_i)

            # SDE Eq: 10
            # u_hat_t_i = - 1 / (1 - t_i) * (Y_t - gamma * y_1)

            # diffusion = torch.sqrt(2 * (1 - gamma)* t_i / (1 - t_i))
            # noise = torch.randn_like(Y_t)
            
            # print((u_hat_t_i * dt).mean(), (torch.sqrt(dt) * diffusion * noise).mean())
            # Y_t = Y_t + u_hat_t_i * dt +  torch.sqrt(dt) * diffusion * noise

            # update Y_t
            Y_t = Y_t + u_hat_t_i * dt # (sigmas[i] - sigmas[i+1])

            # debug print save image 
            # debug_latents = self._unpack_latents(Y_t, height * self.vae_scale_factor // 2, width * self.vae_scale_factor // 2, self.vae_scale_factor)
            # debug_latents = (debug_latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            # debug_image = self.vae.decode(debug_latents, return_dict=False)[0]
            # debug_image = self.image_processor.postprocess(debug_image, output_type='pil')
            # debug_image[0].save(f"debug/forward_{i}.png")

        return Y_t

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt


    def _get_prompt_emb(self, prompt):
        prompt_embeds = self.pipe.encode_prompt(
            prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )[0]
        return prompt_embeds

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: Any = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 0.6,
        eta: float = 1.0,
        gamma: float = 1.0,
        start_timestep: int = 0,
        stop_timestep: int = 6,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        lambda_weight: float = 0.1,
        mask: Optional[torch.Tensor] = None,
        mask_index: int = 0
    ):
        import torchvision.utils as vutils

        height = height or self.args.default_sample_size * self.args.vae_scale_factor
        width = width or self.args.default_sample_size * self.args.vae_scale_factor

        if start_timestep < 0 or start_timestep > stop_timestep:
            raise ValueError(f"`start_timestep` should be in [0, stop_timestep] but is {start_timestep}")

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        self.check_inputs(prompt, prompt_2, strength, height, width,
                          prompt_embeds=prompt_embeds, pooled_prompt_embeds=pooled_prompt_embeds,
                          callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                          max_sequence_length=max_sequence_length)

        if self.edit_prompt is not None:
            self.edit_prompt_emb = self._get_prompt_emb(self.edit_prompt)

        init_image = self.image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)

        batch_size = 1 if isinstance(prompt, str) else len(prompt) if isinstance(prompt, list) else prompt_embeds.shape[0]
        device = self._execution_device

        lora_scale = self._joint_attention_kwargs.get("scale", None) if self._joint_attention_kwargs else None

        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt, prompt_2, prompt_embeds, pooled_prompt_embeds,
            device, num_images_per_prompt, max_sequence_length, lora_scale)

        null_prompt_embeds, null_pooled_prompt_embeds, null_text_ids = self.encode_prompt(
            "", "", None, None,
            device, num_images_per_prompt, max_sequence_length, lora_scale)

        self.for_prompt_emb = prompt_embeds
        self.null_prompt_emb = null_prompt_embeds

        if not hasattr(self, "edit_prompt_emb"):
            self.edit_prompt_emb = None

        xT = torch.randn(1, self.c_in, self.image_size, self.image_size, dtype=self.dtype, device=self.device)

        if self.args.mask_type == "SAM":
            mask_path = os.path.join(self.result_folder, "mask/mask.pt")
            image_path = os.path.join(self.result_folder, "original_stage1.png")

            if not os.path.exists(image_path) or not os.path.exists(mask_path):
                print("Generating images and creating masks......")
                self.EXP_NAME = "original"
                x0 = self.DDPMforwardsteps(
                    xT, t_start_idx=0, t_end_idx=-1,
                    for_prompt_emb=self.for_prompt_emb,
                    edit_prompt_emb=self.edit_prompt_emb,
                    null_prompt_emb=self.null_prompt_emb,
                    mode="null+(for-null)"
                )
                x0_stage2_pil, x0_stage3_pil = self.superresolution(
                    [Image.fromarray(x0[0].detach().cpu().numpy())],
                    self.for_prompt, self.for_prompt_emb, self.null_prompt_emb
                )
                masks = self.sam.mask_segmentation(x0_stage2_pil, resolution=self.image_size)
            else:
                print("Loading masks......")
                masks = torch.load(mask_path)

            mask = masks[mask_index].squeeze(dim=0).repeat(3, 1, 1)

###############################
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = (height // self.vae_scale_factor) * (width // self.vae_scale_factor)
        mu = calculate_shift(image_seq_len, self.scheduler.config.base_image_seq_len,
                            self.scheduler.config.max_image_seq_len,
                            self.scheduler.config.base_shift,
                            self.scheduler.config.max_shift)

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas, mu=mu)
        timesteps, sigmas, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)

        if num_inference_steps < 1:
            raise ValueError(f"Invalid number of steps: {num_inference_steps} for strength: {strength}")

        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        num_channels_latents = self.transformer.config.in_channels // 4

        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32) \
            if self.transformer.config.guidance_embeds else None



        ori_latents, latents, latent_image_ids = self.prepare_latents(
            init_image, latent_timestep, batch_size * num_images_per_prompt, num_channels_latents,
            height, width, prompt_embeds.dtype, device, generator, latents, gamma, sigmas,
            null_prompt_embeds, null_pooled_prompt_embeds, null_text_ids,
            timesteps=timesteps, guidance=guidance)

        if self.transformer.config.guidance_embeds:
            guidance = guidance.expand(latents.shape[0])

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        ################
        #denoising loop#
        ################
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                t_i = 1 - t / 1000
                dt = torch.tensor(1 / (len(timesteps) - 1), device=device)
                if self.interrupt:
                    continue
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                v_t = -self.transformer(
                    hidden_states=latents, timestep=timestep / 1000, guidance=guidance,
                    pooled_projections=pooled_prompt_embeds, encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids, img_ids=latent_image_ids,
                    joint_attention_kwargs=self._joint_attention_kwargs, return_dict=False)[0]

                x_1 = latents + v_t * (1 - t_i)

                y_hat_0 = x_1.clone().detach().requires_grad_(True)
                optimizer = torch.optim.Adam([y_hat_0], lr=0.1)
                for _ in range(10):
                    loss1 = torch.nn.functional.mse_loss(mask * ori_latents, mask * y_hat_0)
                    loss2 = torch.nn.functional.mse_loss(x_1, y_hat_0)
                    loss = loss1 + lambda_weight * loss2
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if i % 5 == 0:
                    images = []
                    for tensor, name in zip([ori_latents, x_1, y_hat_0], ["ori_latents", "x_1", "y_hat_0"]):
                        latent_dec = self._unpack_latents(tensor, height, width, self.vae_scale_factor)
                        latent_dec = (latent_dec / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                        decoded = self.vae.decode(latent_dec, return_dict=False)[0]
                        post = self.image_processor.postprocess(decoded, output_type="pil")
                        images.append(wandb.Image(post[0], caption=name))

                v_t_cond = (y_hat_0 - latents) / (1 - t_i)
                eta_t = eta if start_timestep <= i < stop_timestep else 0.0
                v_hat_t = v_t + eta_t * (v_t_cond - v_t) if start_timestep <= i < stop_timestep else v_t

                latents_dtype = latents.dtype
                latents = latents + v_hat_t * (sigmas[i] - sigmas[i + 1])

                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

                if callback_on_step_end:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image)


if __name__ == "__main__":
    import argparse
    import wandb
    import torch
    from PIL import Image
    from io import BytesIO
    import requests
    from diffusers.utils import load_image
    from pipeline_flux_rf_inversion_m import FluxRFInversionPipeline
    from diffusers import FluxImg2ImgPipeline

    def main():
        import torchvision.transforms as T

        parser = argparse.ArgumentParser(description="Run Flux RF Inversion Pipeline")
        parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-schnell")
        parser.add_argument("--image", type=str, default="https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg")
        parser.add_argument("--prompt", type=str, default="cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k")
        parser.add_argument("--prompt_2", type=str)
        parser.add_argument("--num_inference_steps", type=int, default=28)
        parser.add_argument("--strength", type=float, default=0.95)
        parser.add_argument("--guidance_scale", type=float, default=3.5)
        parser.add_argument("--gamma", type=float, default=0.5)
        parser.add_argument("--eta", type=float, default=0.9)
        parser.add_argument("--start_timestep", type=int, default=0)
        parser.add_argument("--stop_timestep", type=int, default=6)
        parser.add_argument("--output", type=str, default="output.jpg")
        parser.add_argument("--use_img2img", action="store_true")
        parser.add_argument("--num_images", type=int, default=1)
        parser.add_argument("--mask", type=str, help="Path to binary mask image (white = keep, black = edit)")

        args = parser.parse_args()

       # wandb.init(project="rf-inversion-debug", name="cli-run")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if args.use_img2img:
            pipe = FluxImg2ImgPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16, cache_dir="/storage/ice-shared/ae8803che/lmkh3/")
        else:
            pipe = FluxRFInversionPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16,cache_dir="/storage/ice-shared/ae8803che/lmkh3/")

        pipe = pipe.to(device)

        if args.image.startswith("http"):
            init_image = load_image(args.image).resize((1024, 1024))
        else:
            init_image = Image.open(args.image).resize((1024, 1024))

        # Load optional mask
        mask_tensor = None
        if args.mask:
            mask_img = Image.open(args.mask).convert("L").resize((1024, 1024))
            transform = T.Compose([T.ToTensor()])
            mask_tensor = transform(mask_img).unsqueeze(0).to(device)
            mask_tensor = (mask_tensor > 0.5).float()

        prompt_2 = args.prompt_2 if args.prompt_2 else args.prompt
        save_base, ext = args.output.rsplit(".", 1)

        for i in range(args.num_images):
            kwargs = {
                "gamma": args.gamma,
                "eta": args.eta,
                "start_timestep": args.start_timestep,
                "stop_timestep": args.stop_timestep
            } if not args.use_img2img else dict()

            if mask_tensor is not None:
                kwargs["mask"] = mask_tensor

            result = pipe(
                prompt=args.prompt,
                prompt_2=prompt_2,
                image=init_image,
                num_inference_steps=args.num_inference_steps,
                strength=args.strength,
                guidance_scale=args.guidance_scale,
                **kwargs
            )

            result.images[0].save(f"{save_base}_{i}.{ext}")
            print(f"Output image saved as {save_base}_{i}.{ext}")

    main()


#python /home/hice1/lmoukheiber3/rf_inversion/pipeline_flux_rf_inversion_m.py   --prompt "change the cat to have green eyes"   --image "/home/hice1/lmoukheiber3/rf_inversion/examples/cat.jpg"   --output "//home/hice1/lmoukheiber3/rf_inversion/results/result.jpg"   --num_images 2   --guidance_scale 4.5   --use_img2img