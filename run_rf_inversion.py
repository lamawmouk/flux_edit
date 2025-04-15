import argparse
import torch
from PIL import Image
import requests
from io import BytesIO
import os

# Import the pipeline
from pipeline_flux_rf_inversion import RFInversionFluxPipeline

def load_image(url_or_path):
    """
    Utility function to load an image from either a local path or a URL.
    Returns a PIL Image in RGB mode.
    """
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        response = requests.get(url_or_path)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    else:
        return Image.open(url_or_path).convert("RGB")

def main():
    # 1) Parse arguments
    parser = argparse.ArgumentParser(description="Run RFInversionFluxPipeline")
    parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-dev",
                        help="The model name/path to load for the pipeline.")
    parser.add_argument("--image", type=str,
                        default="https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg",
                        help="URL or path to an input image.")
    parser.add_argument("--output", type=str, default="edited_image.jpg",
                        help="Where to save the final edited image.")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Torch device, e.g. 'cuda:0', 'cpu', etc.")
    parser.add_argument("--cache_folder", type=str, default="/storage/ice-shared/ae8803che/lmkh3/",
                        help="Directory for storing downloaded models.")
    parser.add_argument("--image_size", type=int, default=1024,
                        help="Size used in pipeline for image processing.")
    parser.add_argument("--edit_prompt", type=str, default="turn the left side into a woman",
                        help="Text describing the edit to make.")
    parser.add_argument("--num_inversion_steps", type=int, default=10,
                        help="Number of steps for the invert() call.")
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="Gamma factor in the forward ODE for inversion.")
    parser.add_argument("--num_inference_steps", type=int, default=28,
                        help="Number of backward generation steps.")
    parser.add_argument("--start_timestep", type=float, default=0,
                        help="Start fraction of timesteps for the partial update.")
    parser.add_argument("--stop_timestep", type=float, default=0.25,
                        help="Stop fraction of timesteps for the partial update.")
    parser.add_argument("--guidance_scale", type=float, default=3.5,
                        help="Classifier-free guidance scale for the final generation.")
    parser.add_argument("--eta", type=float, default=0.9,
                        help="Eta parameter for RF inversion.")
    args = parser.parse_args()

    # Print parameters
    print(f"Running with parameters:")
    print(f"  Model: {args.model}")
    print(f"  Image: {args.image}")
    print(f"  Output: {args.output}")
    print(f"  Edit prompt: {args.edit_prompt}")
    print(f"  Cache folder: {args.cache_folder}")

    # 2) Load the custom pipeline
    pipe = RFInversionFluxPipeline.from_pretrained(
        pretrained_model_name_or_path=args.model,
        torch_dtype=torch.float16,
        cache_dir=args.cache_folder
    )
    pipe.to(args.device)

    # 3) Load/prepare the initial image
    init_image = load_image(args.image).resize((args.image_size, args.image_size))
    print(f"Loaded image and resized to {args.image_size}x{args.image_size}")

    # 4) Perform the "invert" step
    print("Performing inversion...")
    inverted_latents, image_latents, latent_image_ids = pipe.invert(
        image=init_image,
        num_inversion_steps=args.num_inversion_steps,
        gamma=args.gamma,
        height=args.image_size,
        width=args.image_size,
    )
    print("Inversion complete!")

    # 5) Perform the final call to apply the edit
    print(f"Applying edit with prompt: '{args.edit_prompt}'")
    result = pipe(
        prompt=args.edit_prompt,
        inverted_latents=inverted_latents,
        image_latents=image_latents,
        latent_image_ids=latent_image_ids,
        height=args.image_size,
        width=args.image_size,
        num_inference_steps=args.num_inference_steps,
        start_timestep=args.start_timestep,
        stop_timestep=args.stop_timestep,
        guidance_scale=args.guidance_scale,
        eta=args.eta,
        output_type="pil",
        return_dict=True,
    )

    # 6) Save the final edited image
    edited_image = result.images[0]
    edited_image.save(args.output)
    print(f"Saved edited image to {args.output}")

if __name__ == "__main__":
    main()

# Example usage:
# python run_rf_inversion.py \
#   --model black-forest-labs/FLUX.1-dev \
#   --image "/home/hice1/lmoukheiber3/rf_inversion/examples/twomen.jpg" \
#   --edit_prompt "turn the left side into a woman" \
#   --cache_folder "/storage/ice-shared/ae8803che/lmkh3/" \
#   --num_inversion_steps 10 \
#   --num_inference_steps 28 \
#   --output "edited_image.jpg"
