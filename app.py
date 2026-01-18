#!/usr/bin/env python3
"""
Zero-1-to-3 Novel View Synthesis App
Generates ±30° rotated views from a single image using Stable Zero123.
"""

import os
import sys
import tempfile

# Add current directory to path for pipeline import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import io
import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
from rembg import remove
import gradio as gr
import imageio

# Determine device
if torch.backends.mps.is_available():
    DEVICE = "mps"
    DTYPE = torch.float32  # MPS works better with float32
elif torch.cuda.is_available():
    DEVICE = "cuda"
    DTYPE = torch.float16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

print(f"Using device: {DEVICE}")

# Global pipeline (loaded once)
PIPELINE = None


def load_pipeline():
    """Load the Zero123 pipeline by manually assembling components."""
    global PIPELINE
    if PIPELINE is not None:
        return PIPELINE

    print("Loading Stable Zero123 pipeline...")

    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    from pipeline_zero1to3 import Zero1to3StableDiffusionPipeline, CCProjection

    model_id = "kxic/stable-zero123"

    # Load each component separately
    print("  Loading VAE...")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=DTYPE)

    print("  Loading image encoder...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_id, subfolder="image_encoder", torch_dtype=DTYPE)

    print("  Loading feature extractor...")
    feature_extractor = CLIPImageProcessor.from_pretrained(model_id, subfolder="feature_extractor")

    print("  Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=DTYPE)

    print("  Loading scheduler...")
    scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

    print("  Loading CC projection...")
    cc_projection = CCProjection.from_pretrained(model_id, subfolder="cc_projection")

    # Assemble the pipeline
    PIPELINE = Zero1to3StableDiffusionPipeline(
        vae=vae,
        image_encoder=image_encoder,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=feature_extractor,
        cc_projection=cc_projection,
        requires_safety_checker=False,
    )

    PIPELINE = PIPELINE.to(DEVICE)
    PIPELINE.enable_attention_slicing()

    print("Pipeline loaded successfully!")
    return PIPELINE


def remove_background(image: Image.Image) -> Image.Image:
    """Remove background from image using rembg."""
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    output = remove(image)
    return output


def prepare_for_diffusion(image: Image.Image) -> Image.Image:
    """
    Prepare image for Zero123 by matching input quality to expected output quality.
    Instead of fighting detail loss, we embrace it - slightly soften the input
    so the output feels consistent rather than degraded.
    """
    # Convert to RGB for processing
    if image.mode == "RGBA":
        alpha = image.split()[3]
        rgb = image.convert("RGB")
    else:
        alpha = None
        rgb = image.convert("RGB")

    # Gentle blur to match diffusion output characteristics
    # This makes the original and generated frames feel cohesive
    rgb = rgb.filter(ImageFilter.GaussianBlur(radius=0.5))

    # Slightly reduce contrast to match diffusion tendency
    enhancer = ImageEnhance.Contrast(rgb)
    rgb = enhancer.enhance(0.95)

    # Gentle color boost - diffusion desaturates, so compensate slightly
    enhancer = ImageEnhance.Color(rgb)
    rgb = enhancer.enhance(1.1)

    # Restore alpha if present
    if alpha:
        rgb = rgb.convert("RGBA")
        rgb.putalpha(alpha)

    return rgb


def preprocess_image(image: Image.Image, size: int = 256, input_resolution: int = 512) -> Image.Image:
    """Preprocess image for Zero123: resize, center, add white background."""

    # Early downscale - reduces noise/artifacts and speeds up background removal
    # Also creates a more "painterly" quality that matches diffusion output
    w, h = image.size
    if max(w, h) > input_resolution:
        scale = input_resolution / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    # Remove background
    image = remove_background(image)

    # Prepare image to match diffusion output quality
    image = prepare_for_diffusion(image)

    # Get the bounding box of non-transparent pixels
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)

    # Resize while maintaining aspect ratio
    w, h = image.size
    scale = min(size / w, size / h) * 0.85  # 85% fill for better detail
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Create white background and paste centered
    result = Image.new("RGB", (size, size), (255, 255, 255))
    paste_x = (size - new_w) // 2
    paste_y = (size - new_h) // 2

    if image.mode == "RGBA":
        result.paste(image, (paste_x, paste_y), image)
    else:
        result.paste(image, (paste_x, paste_y))

    return result


def generate_novel_view(processed_image: Image.Image, azimuth: float, polar: float = 0,
                        num_steps: int = 75, guidance: float = 3.0) -> Image.Image:
    """Generate a novel view at the specified angles."""
    pipe = load_pipeline()

    # Zero123 pose format: [polar_deg, azimuth_deg, distance]
    # polar: elevation angle (up/down)
    # azimuth: rotation around vertical axis (left/right)
    pose = [polar, azimuth, 0.0]

    with torch.no_grad():
        result = pipe(
            input_imgs=processed_image,
            prompt_imgs=processed_image,
            poses=[pose],
            height=256,
            width=256,
            num_inference_steps=num_steps,
            guidance_scale=guidance,
        ).images[0]

    return result


def generate_rotation_set(image: Image.Image, angle: float = 30,
                          num_steps: int = 75, guidance: float = 3.0,
                          num_frames: int = 3, black_white: bool = False,
                          input_resolution: int = 512):
    """Generate frames across rotation range from +angle to -angle."""

    # Preprocess the image once (removes background)
    print(f"Preprocessing image (input res: {input_resolution}px)...")
    processed = preprocess_image(image, input_resolution=input_resolution)

    pipe = load_pipeline()

    results = []

    # Generate evenly spaced angles from +angle to -angle
    if num_frames == 1:
        angles = [0]
    else:
        angles = [angle - (2 * angle * i / (num_frames - 1)) for i in range(num_frames)]

    for i, az in enumerate(angles):
        if abs(az) < 0.01:  # Essentially zero
            # For center image, use the preprocessed version directly
            frame = processed
            print(f"Frame {i+1}/{num_frames}: Using original (azimuth=0°)")
        else:
            print(f"Frame {i+1}/{num_frames}: Generating view at azimuth={az:.1f}°...")
            pose = [0, az, 0.0]  # [polar, azimuth, distance]

            with torch.no_grad():
                frame = pipe(
                    input_imgs=processed,
                    prompt_imgs=processed,
                    poses=[pose],
                    height=256,
                    width=256,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance,
                ).images[0]

        # Convert to black and white if enabled
        if black_white:
            frame = frame.convert("L").convert("RGB")

        results.append(frame)

    return results


def create_gif(images: list, duration: float = 0.5, boomerang: bool = False) -> bytes:
    """Create a GIF from a list of PIL images."""
    gif_buffer = io.BytesIO()
    frames = [np.array(img) for img in images]

    # Boomerang: start from center (original), go to end, back to start, back to center
    # For frames [1,2,3] (where 2 is center): becomes [2,3,2,1,2,3,2,1...]
    # For frames [1,2,3,4,5] (where 3 is center): becomes [3,4,5,4,3,2,1,2,3...]
    if boomerang and len(frames) >= 3:
        mid = len(frames) // 2
        # Start at center, go right to end, back through center to start, back to center
        # Build: center->end, end->start (skip center), start->center (skip start)
        frames = frames[mid:] + frames[mid - 1::-1] + frames[1:mid]

    imageio.mimsave(gif_buffer, frames, format='GIF', duration=duration, loop=0)
    gif_buffer.seek(0)
    return gif_buffer.getvalue()


def export_images(images: list, output_dir: str, base_name: str = "view"):
    """Export images to the specified directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for i, img in enumerate(images):
        filename = f"{base_name}_frame_{i:02d}.png"
        filepath = output_path / filename
        img.save(filepath)
        saved_paths.append(str(filepath))
        print(f"Saved: {filepath}")

    # Also save the GIF
    gif_path = output_path / f"{base_name}_animation.gif"
    gif_data = create_gif(images)
    with open(gif_path, "wb") as f:
        f.write(gif_data)
    saved_paths.append(str(gif_path))
    print(f"Saved: {gif_path}")

    return saved_paths


# Store generated images globally for export
CURRENT_IMAGES = []


def process_image(image, angle, num_steps, guidance, num_frames, black_white, boomerang, input_resolution):
    """Main processing function for Gradio."""
    global CURRENT_IMAGES

    if image is None:
        return None, None, "Please upload an image first."

    try:
        # Generate the rotated views
        images = generate_rotation_set(
            Image.fromarray(image),
            angle=angle,
            num_steps=int(num_steps),
            guidance=guidance,
            num_frames=int(num_frames),
            black_white=black_white,
            input_resolution=int(input_resolution)
        )

        CURRENT_IMAGES = images

        # Create GIF preview in system temp directory
        gif_data = create_gif(images, duration=0.5, boomerang=boomerang)
        gif_path = os.path.join(tempfile.gettempdir(), "preview.gif")
        with open(gif_path, "wb") as f:
            f.write(gif_data)

        return images, gif_path, f"Generated {len(images)} frames!"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, f"Error: {str(e)}"


def do_export(output_dir):
    """Export current images to directory."""
    global CURRENT_IMAGES

    if not CURRENT_IMAGES:
        return "No images to export. Generate images first."

    if not output_dir:
        return "Please specify an output directory."

    try:
        saved = export_images(CURRENT_IMAGES, output_dir)
        return f"Exported {len(saved)} files to {output_dir}"
    except Exception as e:
        return f"Export error: {str(e)}"


def create_ui():
    """Create the Gradio interface."""

    # Clean, modern theme with good contrast
    theme = gr.themes.Base(
        primary_hue="blue",
        secondary_hue="gray",
    ).set(
        button_primary_background_fill="#2563eb",
        button_primary_background_fill_hover="#1d4ed8",
        button_primary_text_color="white",
        block_title_text_weight="600",
        block_label_text_weight="500",
    )

    with gr.Blocks(title="Zero-1-to-3 Novel View Synthesis", theme=theme) as app:
        gr.Markdown("# Zero-1-to-3 Novel View Synthesis")
        gr.Markdown("Upload an image to generate rotated views using Stable Zero123.")

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Upload Image", type="numpy")

                with gr.Group():
                    gr.Markdown("### Settings")
                    angle_slider = gr.Slider(
                        minimum=10, maximum=60, value=30, step=5,
                        label="Rotation Angle (±degrees)"
                    )
                    frames_slider = gr.Slider(
                        minimum=3, maximum=12, value=3, step=1,
                        label="Number of Frames"
                    )
                    steps_slider = gr.Slider(
                        minimum=25, maximum=100, value=100, step=5,
                        label="Inference Steps (more = better quality)"
                    )
                    guidance_slider = gr.Slider(
                        minimum=1.0, maximum=10.0, value=4.0, step=0.5,
                        label="Guidance Scale (higher = sharper details)"
                    )
                    bw_checkbox = gr.Checkbox(
                        value=False,
                        label="Black & White"
                    )
                    boomerang_checkbox = gr.Checkbox(
                        value=False,
                        label="Boomerang (1-2-3-2-1)"
                    )
                    input_res_slider = gr.Slider(
                        minimum=256, maximum=1024, value=512, step=128,
                        label="Input Resolution (lower = softer, faster)"
                    )

                generate_btn = gr.Button("Generate Views", variant="primary")

                with gr.Group():
                    gr.Markdown("### Export")
                    output_dir = gr.Textbox(
                        label="Export Directory",
                        placeholder="/path/to/output/folder"
                    )
                    export_btn = gr.Button("Export Images")

                status = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=2):
                gallery = gr.Gallery(label="Generated Frames", columns=4, height="auto")
                gif_preview = gr.Image(label="GIF Preview", type="filepath")

        # Connect events
        generate_btn.click(
            fn=process_image,
            inputs=[input_image, angle_slider, steps_slider, guidance_slider, frames_slider, bw_checkbox, boomerang_checkbox, input_res_slider],
            outputs=[gallery, gif_preview, status]
        )

        export_btn.click(
            fn=do_export,
            inputs=[output_dir],
            outputs=[status]
        )

    return app


def main():
    """Main entry point."""
    print("=" * 50)
    print("Zero-1-to-3 Novel View Synthesis")
    print("=" * 50)
    print(f"Device: {DEVICE}")
    print("Loading model (this may take a moment)...")
    print("=" * 50)

    # Pre-load the pipeline
    load_pipeline()

    print("Starting web UI at http://127.0.0.1:7860")

    # Launch Gradio
    app = create_ui()
    app.launch(share=False, server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
