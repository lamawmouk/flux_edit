# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
# modeled after RF Inversion: https://rf-inversion.github.io/, authored by Litu Rout, Yujia Chen, Nataniel Ruiz,
# Constantine Caramanis, Sanjay Shakkottai and Wen-Sheng Chu.
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

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
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


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> import requests
        >>> import PIL
        >>> from io import BytesIO
        >>> from diffusers import DiffusionPipeline

        >>> pipe = DiffusionPipeline.from_pretrained(
        ...    "black-forest-labs/FLUX.1-dev",
        ...    torch_dtype=torch.bfloat16,
        ...    custom_pipeline="pipeline_flux_rf_inversion")
        >>> pipe.to("cuda")

         >>> def download_image(url):
        ...     response = requests.get(url)
        ...     return PIL.Image.open(BytesIO(response.content)).convert("RGB")


        >>> img_url = "https://www.aiml.informatik.tu-darmstadt.de/people/mbrack/tennis.jpg"
        >>> image = download_image(img_url)

        >>> inverted_latents, image_latents, latent_image_ids = pipe.invert(image=image, num_inversion_steps=28, gamma=0.5)

        >>> edited_image = pipe(
        ...     prompt="a tomato",
        ...     inverted_latents=inverted_latents,
        ...     image_latents=image_latents,
        ...     latent_image_ids=latent_image_ids,
        ...     start_timestep=0,
        ...     stop_timestep=.25,
        ...     num_inference_steps=28,
        ...     eta=0.9,
        ... ).images[0]
        ```
"""


# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
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


class RFInversionFluxPipeline(
    DiffusionPipeline,
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin,
):
    r"""
    The Flux pipeline for text-to-image generation.

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

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
        mask_type="SAM",
        args
        
       
  
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
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = 128
        
        #####added#######
        self.seed = args.seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self._execution_device = self.device
        self.default_sample_size = 64
        self.c_in = 4
        self.mask_type = args.mask_type
        if self.mask_type == "SAM":
            self.sam = SAM(args, log_dir = self.result_folder)
         self.result_folder = os.path.join(args.result_folder, f"for_prompt_{args.for_prompt}_seed{args.seed}")
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)



    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._get_t5_prompt_embeds
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer_2)

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

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

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

    @torch.no_grad()
    # Modified from diffusers.pipelines.ledits_pp.pipeline_leditspp_stable_diffusion.LEditsPPPipelineStableDiffusion.encode_image
    def encode_image(self, image, dtype=None, height=None, width=None, resize_mode="default", crops_coords=None):
        image = self.image_processor.preprocess(
            image=image, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords
        )
        resized = self.image_processor.postprocess(image=image, output_type="pil")

        if max(image.shape[-2:]) > self.vae.config["sample_size"] * 1.5:
            logger.warning(
                "Your input images far exceed the default resolution of the underlying diffusion model. "
                "The output images may contain severe artifacts! "
                "Consider down-sampling the input using the `height` and `width` parameters"
            )
        image = image.to(dtype)

        x0 = self.vae.encode(image.to(self._execution_device)).latent_dist.sample()
        x0 = (x0 - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        x0 = x0.to(dtype)
        return x0, resized

    def check_inputs(
        self,
        prompt,
        prompt_2,
        inverted_latents,
        image_latents,
        latent_image_ids,
        height,
        width,
        start_timestep,
        stop_timestep,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor} but are {height} and {width}."
            )

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

        if inverted_latents is not None and (image_latents is None or latent_image_ids is None):
            raise ValueError(
                "If `inverted_latents` are provided, `image_latents` and `latent_image_ids` also have to be passed. "
            )
        # check start_timestep and stop_timestep
        if start_timestep < 0 or start_timestep > stop_timestep:
            raise ValueError(f"`start_timestep` should be in [0, stop_timestep] but is {start_timestep}")

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        height = height // vae_scale_factor
        width = width // vae_scale_factor

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def prepare_latents_inversion(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        image_latents,
    ):
        height = int(height) // self.vae_scale_factor
        width = int(width) // self.vae_scale_factor

        latents = self._pack_latents(image_latents, batch_size, num_channels_latents, height, width)

        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        return latents, latent_image_ids

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline.prepare_latents
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        return latents, latent_image_ids

    # Copied from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_img2img.StableDiffusion3Img2ImgPipeline.get_timesteps
    def get_timesteps(self, num_inference_steps, strength=1.0):
        # get the original timestep using init_timestep
        init_timestep = min(num_inference_steps * strength, num_inference_steps)

        t_start = int(max(num_inference_steps - init_timestep, 0))
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        sigmas = self.scheduler.sigmas[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, sigmas, num_inference_steps - t_start

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
    # utils
    def _get_prompt_emb(self, prompt):
        prompt_embeds = self.encode_prompt(
            prompt,
            device = self.device,
            num_images_per_prompt = 1,
            do_classifier_free_guidance = False,
        )[0]
        return prompt_embeds
    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)

# Patched and refactored __call__ method section
# Assumes this is inside a class with proper attributes defined


    # ========== Utility ==========
    # Encode a prompt using the underlying pipeline's encode_prompt method
    def _get_prompt_emb(self, prompt):
        prompt_embeds = self.pipe.encode_prompt(
            prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )[0]
        return prompt_embeds

    # Compute y_hat minimizing masked error + latent similarity
    def _compute_y_hat(self, image_latents, x_1, mask, lambda_reg=1.0):
        masked_y0 = mask * image_latents
        y_hat = (masked_y0 + lambda_reg * x_1) / (mask + lambda_reg + 1e-6)
        return y_hat

    # ========== Main Inference Call ==========
    def __call__(
        self,
        edit_prompt: str,
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        inverted_latents: Optional[torch.FloatTensor] = None,
        image_latents: Optional[torch.FloatTensor] = None,
        latent_image_ids: Optional[torch.FloatTensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        eta: float = 1.0,
        decay_eta: Optional[bool] = False,
        eta_decay_power: Optional[float] = 1.0,
        strength: float = 1.0,
        start_timestep: float = 0,
        stop_timestep: float = 0.25,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        timesteps: List[int] = None,
        guidance_scale: float = 3.5,
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
        image: Optional[torch.FloatTensor] = None,
        mask_index: int = 0
    ):
        # ========== Defaults and Input Checking ==========
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self.check_inputs(
            prompt,
            prompt_2,
            inverted_latents,
            image_latents,
            latent_image_ids,
            height,
            width,
            start_timestep,
            stop_timestep,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        # ========== Init State ==========
        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        do_rf_inversion = inverted_latents is not None

        # ========== Determine Batch Size ==========
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # ========== Encode Prompts ==========
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # ========== Prepare Latents ==========
        num_channels_latents = self.transformer.config.in_channels // 4
        if do_rf_inversion:
            latents = inverted_latents
        else:
            latents, latent_image_ids = self.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

        # ========== Prepare Timesteps ==========
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = (int(height) // self.vae_scale_factor // 2) * (int(width) // self.vae_scale_factor // 2)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        if do_rf_inversion:
            start_timestep = int(start_timestep * num_inference_steps)
            stop_timestep = min(int(stop_timestep * num_inference_steps), num_inference_steps)
            timesteps, sigmas, num_inference_steps = self.get_timesteps(num_inference_steps, strength)

        # ========== Final Scheduler State ==========
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # ========== Optional Guidance Embedding ==========
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # ========== Required Edit Prompt Embedding ==========
        self.edit_prompt = edit_prompt
        self.edit_prompt_emb = self._get_prompt_emb(edit_prompt)

        # ========== Prepare Init Image ==========
        init_image = self.image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)

        # ========== Redundant Prompt Setup (TODO: Remove?) ==========
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

        # ========== Random Noise for Denoising ==========
        xT = torch.randn(1, self.c_in, self.image_size, self.image_size, dtype=self.dtype, device=self.device)

        # ========== SAM Masking (Required) ==========
        assert self.args.mask_type == "SAM", "Only SAM masking is supported."
        # Perform SAM masking
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

        # ========== Denoising Loop / Controlled Reverse ODE ==========
        # Additional computation example: x₁ = xₜ + v(xₜ, t, edit_prompt) * (1 - t)
        # Use edit prompt embedding and latents to compute velocity
        with torch.no_grad():
            timestep_scalar = timesteps[0] / 1000  # example timestep for demo
            timestep_tensor = timestep_scalar.expand(latents.shape[0]).to(latents.dtype)
            v_xt = self.transformer(
                hidden_states=latents,
                timestep=timestep_tensor,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=self.edit_prompt_emb,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=self._joint_attention_kwargs,
                return_dict=False,
            )[0]
            x_1 = latents + v_xt * (1 - timestep_scalar)
        # Compute ŷ_hat from x₁ and image_latents using helper
        y_hat = self._compute_y_hat(image_latents, x_1, mask)

        # Run iterative denoising loop with velocity or scheduler update per timestep
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if do_rf_inversion:
                                    # Normalize timestep for reverse ODE
                t_i = 1 - t / 1000
                                    # Compute fixed time step size for continuous update (not used directly)
                dt = torch.tensor(1 / (len(timesteps) - 1), device=device)

                if self.interrupt:
                    continue

                                # Expand scalar timestep to match batch shape
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                                # Track original latent dtype to restore after processing
                latents_dtype = latents.dtype
                if do_rf_inversion:
                                        # Reverse velocity from noise prediction (ODE derivative)
                    v_t = -noise_pred
                                        # Conditional velocity from target (y_hat) direction
                    v_t_cond = (y_hat - latents) / (1 - t_i + 1e-6)
                                        # Apply eta weighting only within specified time window
                    eta_t = eta if start_timestep <= i < stop_timestep else 0.0
                    if decay_eta:
                                                # Decay eta over time if enabled
                        eta_t = eta_t * (1 - i / num_inference_steps) ** eta_decay_power
                                        # Final weighted velocity update
                    v_hat_t = v_t + eta_t * (v_t_cond - v_t)
                                        # Integrate latent update using differential step
                    latents = latents + v_hat_t * (sigmas[i] - sigmas[i + 1])
                else:
                                        # Use built-in scheduler to update latents in non-inversion case
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                                                # Patch dtype mismatch (e.g. MPS bug workaround)
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                                        # Prepare callback input values dynamically by name
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                                        # Update visual progress bar if conditions match
                    progress_bar.update()

                if XLA_AVAILABLE:
                    # XLA optimization step for TPU/multi-core execution
                    xm.mark_step()

        # ========== Output Handling ==========
        if output_type == "latent":
            image = latents  # Return raw latents if requested
        else:
            # Decode latents to image space using VAE
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload models/hooks
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return FluxPipelineOutput(images=image, y_hat=y_hat)


    @torch.no_grad()
    def invert(
        self,
        image: PipelineImageInput,
        source_prompt: str = "",
        source_guidance_scale=0.0,
        num_inversion_steps: int = 28,
        strength: float = 1.0,
        gamma: float = 0.5,
        height: Optional[int] = None,
        width: Optional[int] = None,
        timesteps: List[int] = None,
        dtype: Optional[torch.dtype] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        r"""
        Performs Algorithm 1: Controlled Forward ODE from https://arxiv.org/pdf/2410.10792
        Args:
            image (`PipelineImageInput`):
                Input for the image(s) that are to be edited. Multiple input images have to default to the same aspect
                ratio.
            source_prompt (`str` or `List[str]`, *optional* defaults to an empty prompt as done in the original paper):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            source_guidance_scale (`float`, *optional*, defaults to 0.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). For this algorithm, it's better to keep it 0.
            num_inversion_steps (`int`, *optional*, defaults to 28):
                The number of discretization steps.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            gamma (`float`, *optional*, defaults to 0.5):
                The controller guidance for the forward ODE, balancing faithfulness & editability:
                higher eta - better faithfullness, less editability. For more significant edits, lower the value of eta.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
        """
        dtype = dtype or self.text_encoder.dtype
        batch_size = 1
        self._joint_attention_kwargs = joint_attention_kwargs
        num_channels_latents = self.transformer.config.in_channels // 4

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        device = self._execution_device

        # 1. prepare image
        image_latents, _ = self.encode_image(image, height=height, width=width, dtype=dtype)
        image_latents, latent_image_ids = self.prepare_latents_inversion(
            batch_size, num_channels_latents, height, width, dtype, device, image_latents
        )

        # 2. prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inversion_steps, num_inversion_steps)
        image_seq_len = (int(height) // self.vae_scale_factor // 2) * (int(width) // self.vae_scale_factor // 2)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inversion_steps = retrieve_timesteps(
            self.scheduler,
            num_inversion_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        timesteps, sigmas, num_inversion_steps = self.get_timesteps(num_inversion_steps, strength)

        # 3. prepare text embeddings
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=source_prompt,
            prompt_2=source_prompt,
            device=device,
        )
        # 4. handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], source_guidance_scale, device=device, dtype=torch.float32)
        else:
            guidance = None

        # Eq 8 dY_t = [u_t(Y_t) + γ(u_t(Y_t|y_1) - u_t(Y_t))]dt
        Y_t = image_latents
        y_1 = torch.randn_like(Y_t)
        N = len(sigmas)

        # forward ODE loop
        with self.progress_bar(total=N - 1) as progress_bar:
            for i in range(N - 1):
                t_i = torch.tensor(i / (N), dtype=Y_t.dtype, device=device)
                timestep = torch.tensor(t_i, dtype=Y_t.dtype, device=device).repeat(batch_size)

                # get the unconditional vector field
                u_t_i = self.transformer(
                    hidden_states=Y_t,
                    timestep=timestep,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # get the conditional vector field
                u_t_i_cond = (y_1 - Y_t) / (1 - t_i)

                # controlled vector field
                # Eq 8 dY_t = [u_t(Y_t) + γ(u_t(Y_t|y_1) - u_t(Y_t))]dt
                u_hat_t_i = u_t_i + gamma * (u_t_i_cond - u_t_i)
                Y_t = Y_t + u_hat_t_i * (sigmas[i] - sigmas[i + 1])
                progress_bar.update()

        # return the inverted latents (start point for the denoising loop), encoded image & latent image ids
        return Y_t, image_latents, latent_image_ids