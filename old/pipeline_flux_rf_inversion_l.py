import os
import gc
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn.functional as F
import torchvision.utils as tvu
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ----------- Transformers / Diffusers -----------
from transformers import (
    pipeline,               # used in SAM
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast
)
from diffusers import (
    DDIMScheduler,
    FlowMatchEulerDiscreteScheduler
)
from diffusers.loaders import (
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin
)
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import (
    pt_to_pil,               # For converting tensors -> PIL
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
    randn_tensor
)
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


################################################################################
# SAM class
################################################################################
class SAM:
    """
    A minimal example of a mask-generation pipeline using a huggingface transformer.
    Assumes your 'args' object has attributes:
      - mask_model_name (str): The model to load for mask-generation
      - device (str/int): e.g. "cuda:0"
      - cache_folder (str): local path for caching model
      - filter_mask (int): threshold for ignoring small masks
    """
    def __init__(self, args, log_dir):
        self.generator = pipeline(
            "mask-generation",
            model=args.mask_model_name,
            device=args.device,
            torch_dtype=torch.float32,
            cache_dir=args.cache_folder,
        )
        self.args = args
        self.log_dir = os.path.join(log_dir, "mask")
        self.transparency = 0.4

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def mask_segmentation(self, image: Image.Image, resolution: int = 64):
        """
        Generate segmentation masks for a PIL image, then resize them to (resolution x resolution).
        Saves intermediate mask images to disk, returns a [N, resolution, resolution] bool tensor.
        """
        outputs = self.generator(image, points_per_batch=64)
        masks = outputs["masks"]  # shape [num_masks, H, W]

        # For debugging or visualization:
        self.show_masks_on_image(image, masks)

        # Convert to torch, downsample
        masks = torch.tensor(masks, dtype=torch.float32)  # [N,H,W] float
        masks = F.interpolate(
            masks.unsqueeze(1), (resolution, resolution), mode="nearest"
        ).squeeze(1)  # [N,res,res]
        masks = torch.round(masks).bool()  # to {0,1} as bool

        # Save
        torch.save(masks, os.path.join(self.log_dir, "mask.pt"))
        return masks

    def color_mask(self, mask: np.ndarray, random_color=False):
        """
        Create an RGB color overlay for a 2D mask in shape [H,W].
        If random_color=True, picks a random color.
        """
        if random_color:
            color = np.random.rand(3) * 255
        else:
            color = np.array([30, 144, 255])  # example color

        h, w = mask.shape[-2:]
        color_mask = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        return color_mask

    def show_masks_on_image(self, raw_image: Image.Image, masks: np.ndarray):
        """
        Overlays each mask (from 'masks') onto raw_image with partial transparency,
        saving each individual mask overlay and a final combined overlay.
        """
        image = np.array(raw_image)
        image_totalmask = image.copy()

        for idx, mask in enumerate(masks):
            if mask.sum() > self.args.filter_mask:
                color_mask = self.color_mask(mask, random_color=True)

                image_per_mask = image.copy()
                # Weighted blend for the areas covered by 'mask'
                blend_part = (
                    image_per_mask[mask].astype(np.float32) * (1 - self.transparency)
                    + color_mask[mask].astype(np.float32) * self.transparency
                ) / 2
                image_per_mask[mask] = blend_part.astype(np.uint8)

                blend_total = (
                    image_totalmask[mask].astype(np.float32) * (1 - self.transparency)
                    + color_mask[mask].astype(np.float32) * self.transparency
                ) / 2
                image_totalmask[mask] = blend_total.astype(np.uint8)

                # Save each mask
                outpath = os.path.join(self.log_dir, f"mask_{idx}.png")
                plt.imsave(outpath, image_per_mask)

        # Save combined result
        total_path = os.path.join(self.log_dir, "total_mask.png")
        plt.imsave(total_path, image_totalmask)


################################################################################
# Helper: Concatenate PIL images horizontally
################################################################################
def concatenate_pil_horizontally(pils: List[Image.Image]) -> Image.Image:
    """
    Concatenate a list of PIL images side-by-side into a single row image.
    """
    widths, heights = zip(*(img.size for img in pils))
    total_width = sum(widths)
    max_height = max(heights)

    new_image = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in pils:
        new_image.paste(img, (x_offset, 0))
        x_offset += img.width

    return new_image


################################################################################
# Timesteps Helpers
################################################################################
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    """
    Utility used by FlowMatchEulerDiscreteScheduler for shifting depending on image_seq_len.
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls scheduler.set_timesteps(...) to retrieve a properly spaced timesteps list.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")

    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(f"Scheduler {scheduler.__class__} does not support custom timesteps.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)

    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(f"Scheduler {scheduler.__class__} does not support custom sigmas.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)

    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

    return timesteps, num_inference_steps


################################################################################
# The Main RFInversionFluxPipeline
################################################################################
class FluxRFInversionPipeline(
    DiffusionPipeline,
    FluxLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin,
):
    """
    A custom pipeline implementing RF Inversion logic on top of a Flux pipeline.
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
        args=None,
    ):
        super().__init__()
        # Register submodules for saving/loading
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        if tokenizer is not None:
            self.tokenizer_max_length = tokenizer.model_max_length
        else:
            self.tokenizer_max_length = 77

        self.default_sample_size = 64

        # Store or define 'args'
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self._execution_device = self.device

        # Access attributes from args if present
        self.seed = getattr(args, "seed", 42)
        self.mask_type = getattr(args, "mask_type", mask_type)
        self.image_size = getattr(args, "image_size", 512)
        self.for_prompt = getattr(args, "for_prompt", "some prompt")
        self.result_folder = getattr(args, "result_folder", "results")
        os.makedirs(self.result_folder, exist_ok=True)

        if self.mask_type == "SAM":
            if "SAM" not in globals():
                raise ImportError("`SAM` class not found! Provide or remove references to it.")
            self.sam = SAM(args, log_dir=self.result_folder)

        # Additional flags used by your custom code
        self.use_yh_custom_scheduler = False
        self.memory_bound = 1
        self.buffer_device = "cpu"
        self.guidance_scale = 1.0
        self.for_steps = 50  # used by DDPMforwardsteps

    # -------------------------------------------------------------------------
    # Prompt Embeddings
    # -------------------------------------------------------------------------
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)

        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(f"T5 prompt was truncated: {removed_text}")

        prompt_embeds = self.text_encoder_2(text_input_ids, output_hidden_states=False)[0]
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder_2.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        return prompt_embeds

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
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)

        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(f"CLIP prompt was truncated: {removed_text}")

        clip_out = self.text_encoder(text_input_ids, output_hidden_states=False)
        # Use pooled output
        prompt_embeds = clip_out.pooler_output.to(dtype=self.text_encoder.dtype, device=device)

        # Duplicate
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)
        return prompt_embeds

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
        device = device or self._execution_device

        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            # CLIP
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
            # T5
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        # Unscale LoRA
        if self.text_encoder is not None and isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
            unscale_lora_layers(self.text_encoder, lora_scale)
        if self.text_encoder_2 is not None and isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
            unscale_lora_layers(self.text_encoder_2, lora_scale)

        dtype = self.text_encoder.dtype if self.text_encoder else self.transformer.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=dtype)
        return prompt_embeds, pooled_prompt_embeds, text_ids

    @torch.no_grad()
    def encode_image(self, image, dtype=None, height=None, width=None, resize_mode="default", crops_coords=None):
        """
        Encode a PIL/tensor image into latent space via self.vae, returning the latents.
        """
        image = self.image_processor.preprocess(
            image=image, height=height, width=width, resize_mode=resize_mode, crops_coords=crops_coords
        )
        resized = self.image_processor.postprocess(image=image, output_type="pil")

        if max(image.shape[-2:]) > self.vae.config["sample_size"] * 1.5:
            logger.warning("Input image is large, may cause artifacts.")

        if dtype is not None:
            image = image.to(dtype)

        x0 = self.vae.encode(image.to(self._execution_device)).latent_dist.sample()
        # shift + scale
        x0 = (x0 - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return x0.to(dtype), resized

    # -------------------------------------------------------------------------
    # Checking & utilities
    # -------------------------------------------------------------------------
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
                f"`height` and `width` must be divisible by {self.vae_scale_factor}. Got {height} x {width}."
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot provide both `prompt` and `prompt_embeds`.")
        if prompt_2 is not None and prompt_embeds is not None:
            raise ValueError("Cannot provide both `prompt_2` and `prompt_embeds`.")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        if prompt is not None and not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` must be str or list, got {type(prompt)}")
        if prompt_2 is not None and not isinstance(prompt_2, (str, list)):
            raise ValueError(f"`prompt_2` must be str or list, got {type(prompt_2)}")

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError("If `prompt_embeds` is provided, `pooled_prompt_embeds` must be as well.")

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot exceed 512, got {max_sequence_length}.")

        if inverted_latents is not None and (image_latents is None or latent_image_ids is None):
            raise ValueError("Must provide `image_latents` & `latent_image_ids` if `inverted_latents` is used.")

        if start_timestep < 0 or start_timestep > stop_timestep:
            raise ValueError(f"`start_timestep` must be in [0, {stop_timestep}], got {start_timestep}.")

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] += torch.arange(height)[:, None]
        latent_image_ids[..., 2] += torch.arange(width)[None, :]
        h, w, c = latent_image_ids.shape
        latent_image_ids = latent_image_ids.view(h * w, c)
        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        """
        Reshape latents into a "packed" format for the FluxTransformer2DModel (which wants patches).
        """
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        """
        Undo the patch-based shape back to normal [B, C, H, W].
        """
        batch_size, num_patches, channels = latents.shape
        height = height // vae_scale_factor
        width = width // vae_scale_factor

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // 4, height, width)
        return latents

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        self.vae.disable_tiling()

    def prepare_latents_inversion(self, batch_size, num_channels_latents, height, width, dtype, device, image_latents):
        """
        For the `invert` method, pack the image_latents into the shape suitable for the Flux transformer.
        """
        height = int(height) // self.vae_scale_factor
        width = int(width) // self.vae_scale_factor
        latents = self._pack_latents(image_latents, batch_size, num_channels_latents, height, width)
        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
        return latents, latent_image_ids

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
        # Ensure height/width are multiples of 2*(vae_scale_factor)
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError("List of generators length mismatch with batch_size.")

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
        return latents, latent_image_ids

    def get_timesteps(self, num_inference_steps, strength=1.0):
        """
        For a partially used schedule: start at step ~ (1 - strength) * num_inference_steps
        """
        init_timestep = min(num_inference_steps * strength, num_inference_steps)
        t_start = int(max(num_inference_steps - init_timestep, 0))
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        sigmas = self.scheduler.sigmas[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)
        return timesteps, sigmas, num_inference_steps - t_start

    @property
    def num_timesteps(self):
        return getattr(self, "_num_timesteps", 0)

    @property
    def interrupt(self):
        return getattr(self, "_interrupt", False)

    # Simple convenience method
    def _get_prompt_emb(self, prompt):
        emb, _, _ = self.encode_prompt(prompt, None, device=self.device, num_images_per_prompt=1)
        return emb

    # ----------------------------------------------------------------------
    # Additional user-provided methods
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def superresolution(self, x0, cond_prompt_emb, uncond_prompt_emb):
        """
        Example stage-2 upscaling. If you have a real model in `self.stage_2`,
        you can do something like:
            self.stage_2(image=..., prompt_embeds=..., negative_prompt_embeds=...)
        Then save the results. This is just a placeholder logic.
        """
        bs = len(x0)
        # If you have a real pipeline or model called self.stage_2, do:
        # Example:
        stage_2_output = self.stage_2(
            image=x0,
            prompt_embeds=cond_prompt_emb.repeat(bs, 1, 1),
            negative_prompt_embeds=uncond_prompt_emb.repeat(bs, 1, 1),
            output_type="pt",
        ).images

        stage_2_pil = pt_to_pil(stage_2_output)
        out_path = os.path.join(self.result_folder, f'{self.EXP_NAME}_stage2.png')
        concatenate_pil_horizontally(stage_2_pil).save(out_path)

        # stage_3 or more advanced steps can go here
        # Return 2 images or None if not used
        return stage_2_pil[0], None

    @torch.no_grad()
    def DDPMforwardsteps(
        self,
        xt: torch.Tensor,
        t_start_idx: int,
        t_end_idx: int,
        for_prompt_emb,
        edit_prompt_emb,
        null_prompt_emb,
        mode="null+(for-null)",
    ):
        """
        Example forward ODE or forward diffusion approach for "RF Inversion" style. 
        You might use a standard DDIM or DDPM schedule to transform xT -> x0.
        """
        assert mode in ["null+(for-null)+(edit-null)", "null+(for-null)", "null+(edit-null)"]

        num_inference_steps = self.for_steps
        do_classifier_free_guidance = self.guidance_scale > 1.0
        memory_bound = self.memory_bound // 2 if do_classifier_free_guidance else self.memory_bound

        if self.use_yh_custom_scheduler:
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        else:
            self.scheduler = DDIMScheduler.from_config(self.scheduler.config)
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        for t_idx, t in enumerate(self.scheduler.timesteps):
            if t_idx < t_start_idx:
                continue
            elif t_idx == t_start_idx:
                pass
            if t_idx == t_end_idx:
                return xt, t, t_idx

            # chunk if large
            xt = xt.to(self.buffer_device)
            if xt.size(0) == 1:
                xt_buffer = [xt]
            else:
                xt_buffer = list(xt.chunk(xt.size(0) // memory_bound))

            for buffer_idx, chunk in enumerate(xt_buffer):
                chunk = chunk.to(self.device)
                chunk = self.scheduler.scale_model_input(chunk, t)

                noise_pred = self._classifer_free_guidance(
                    chunk,
                    t,
                    for_prompt_emb,
                    edit_prompt_emb,
                    null_prompt_emb,
                    mode=mode,
                    do_classifier_free_guidance=do_classifier_free_guidance
                )
                step_out = self.scheduler.step(noise_pred, t, chunk, eta=0)
                chunk = step_out.prev_sample
                xt_buffer[buffer_idx] = chunk.to(self.buffer_device)

            xt = torch.cat(xt_buffer, dim=0).to(self.device)
            del xt_buffer
            torch.cuda.empty_cache()

        # If we exhaust the loop
        xt = (xt / 2 + 0.5).clamp(0, 1)
        save_path = os.path.join(self.result_folder, f'{self.EXP_NAME}_stage1.png')
        tvu.save_image(xt, save_path, nrow=xt.size(0))
        xt = (xt * 255).to(torch.uint8).permute(0, 2, 3, 1)
        return xt

    def _classifer_free_guidance(
        self,
        xt,
        t,
        for_prompt_emb,
        edit_prompt_emb,
        null_prompt_emb,
        mode="null+(for-null)",
        do_classifier_free_guidance=False,
    ):
        """
        A placeholder for mixing unconditional + conditional prompts, typical in CFG. 
        Replace with actual logic for your pipeline's model forward pass.
        """
        # Example dummy
        noise_pred = torch.zeros_like(xt)
        return noise_pred

    # ----------------------------------------------------------------------
    # Main __call__ method
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def __call__(
        self,
        edit_prompt: str,
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
        start_timestep: float = 0.0,
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
        """
        Example forward call method that might apply partial "RF Inversion" logic or a standard generation approach.
        """
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

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        do_rf_inversion = inverted_latents is not None

        # Determine batch
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0] if prompt_embeds is not None else 1

        device = self._execution_device

        # Encode prompts
        lora_scale = None
        if joint_attention_kwargs is not None:
            lora_scale = joint_attention_kwargs.get("scale", None)

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

        # Prepare latents
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

        # Timesteps
        if sigmas is None:
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)

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
        self._num_timesteps = len(timesteps)

        # Example usage if guidance is used
        if self.transformer.config.guidance_embeds:
            guidance_tensor = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance_tensor = guidance_tensor.expand(latents.shape[0])
        else:
            guidance_tensor = None

        # We might want an edit_prompt embedding
        edit_prompt_emb = self._get_prompt_emb(edit_prompt)

        # If an init image is provided
        if image is not None:
            init_image = self.image_processor.preprocess(
                image, height=height, width=width
            ).to(torch.float32)
        else:
            init_image = None

        # Example partial usage of SAM
        if self.mask_type == "SAM":
            mask_path = os.path.join(self.result_folder, "mask/mask.pt")
            image_path = os.path.join(self.result_folder, "original_stage1.png")

            if not os.path.exists(mask_path):
                print("Generating images & creating masks with SAM...")
                self.EXP_NAME = "original"
                xT = torch.randn(1, 4, self.image_size, self.image_size, dtype=self.dtype, device=self.device)

                # Example custom method forward
                x0, _, _ = self.DDPMforwardsteps(
                    xT,
                    t_start_idx=0,
                    t_end_idx=-1,
                    for_prompt_emb=prompt_embeds,
                    edit_prompt_emb=edit_prompt_emb,
                    null_prompt_emb=None,
                    mode="null+(for-null)",
                )
                # superresolution
                x0_stage2_pil, x0_stage3_pil = self.superresolution(
                    x0, cond_prompt_emb=prompt_embeds, uncond_prompt_emb=None
                )

                # run SAM
                masks = self.sam.mask_segmentation(x0_stage2_pil, resolution=self.image_size)
            else:
                print("Loading existing mask from disk...")
                masks = torch.load(mask_path)

            # pick one mask
            mask = masks[mask_index].squeeze(dim=0).repeat(3, 1, 1)
          

        # After your main logic, decode latents
        if output_type == "latent":
            image_out = latents
        else:
            latents_unpacked = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents_unpacked = (latents_unpacked / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image_out = self.vae.decode(latents_unpacked, return_dict=False)[0]
            image_out = self.image_processor.postprocess(image_out, output_type=output_type)

        if not return_dict:
            return (image_out,)

        return FluxPipelineOutput(images=image_out, y_hat=None)

    # ----------------------------------------------------------------------
    # Inversion method
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def invert(
        self,
        image: PipelineImageInput,
        source_prompt: str = "",
        source_guidance_scale: float = 0.0,
        num_inversion_steps: int = 28,
        strength: float = 1.0,
        gamma: float = 0.5,
        height: Optional[int] = None,
        width: Optional[int] = None,
        timesteps: List[int] = None,
        dtype: Optional[torch.dtype] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Example "inversion" method that attempts a controlled forward ODE approach.
        Returns (inverted_latents, image_latents, latent_image_ids).
        """
        dtype = dtype or self.text_encoder.dtype
        batch_size = 1
        self._joint_attention_kwargs = joint_attention_kwargs

        num_channels_latents = self.transformer.config.in_channels // 4
        height = height or (self.default_sample_size * self.vae_scale_factor)
        width = width or (self.default_sample_size * self.vae_scale_factor)
        device = self._execution_device

        # 1. encode input image
        image_latents, _ = self.encode_image(image, height=height, width=width, dtype=dtype)
        latents, latent_image_ids = self.prepare_latents_inversion(
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
            num_inference_steps=num_inversion_steps,
            device=device,
            timesteps=timesteps,
            sigmas=sigmas,
            mu=mu,
        )
        # partially used schedule
        timesteps, sigmas_, num_inversion_steps = self.get_timesteps(num_inversion_steps, strength)

        # 3. encode prompt
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt=source_prompt,
            prompt_2=source_prompt,
            device=device,
        )
        # guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], source_guidance_scale, device=device, dtype=torch.float32)
        else:
            guidance = None

        # 4. forward ODE
        Y_t = latents
        y_1 = torch.randn_like(Y_t)  # random target
        N = len(sigmas)

        with self.progress_bar(total=N - 1) as progress_bar:
            for i in range(N - 1):
                t_i = torch.tensor(i / N, dtype=Y_t.dtype, device=device)
                timestep = torch.tensor(t_i, dtype=Y_t.dtype, device=device).repeat(batch_size)

                # unconditional vector field
                u_t_i = self.transformer(
                    hidden_states=Y_t,
                    timestep=timestep,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self._joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # conditional field
                u_t_i_cond = (y_1 - Y_t) / (1 - t_i)

                # Controlled forward ODE
                u_hat_t_i = u_t_i + gamma * (u_t_i_cond - u_t_i)
                Y_t = Y_t + u_hat_t_i * (sigmas[i] - sigmas[i + 1])
                progress_bar.update()

        # Return the latents = start point for backward generation, plus the raw image latents, plus IDs
        return Y_t, image_latents, latent_image_ids
