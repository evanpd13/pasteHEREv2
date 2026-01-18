#!/usr/bin/env python3
"""
Zero-1-to-3 Novel View Synthesis App
Generates ±30° rotated views from a single image using Stable Zero123.
"""

import argparse
import os
import sys
import tempfile

# Add current directory to path for pipeline import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import io
import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
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


def prepare_for_diffusion(image: Image.Image,
                          soften_radius: float = 0.5,
                          contrast: float = 0.95,
                          color: float = 1.1,
                          auto_contrast: bool = True) -> Image.Image:
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
    if soften_radius > 0:
        rgb = rgb.filter(ImageFilter.GaussianBlur(radius=soften_radius))

    # Slightly reduce contrast to match diffusion tendency
    enhancer = ImageEnhance.Contrast(rgb)
    rgb = enhancer.enhance(contrast)

    # Gentle color boost - diffusion desaturates, so compensate slightly
    enhancer = ImageEnhance.Color(rgb)
    rgb = enhancer.enhance(color)

    if auto_contrast:
        rgb = ImageOps.autocontrast(rgb, cutoff=1)

    # Restore alpha if present
    if alpha:
        rgb = rgb.convert("RGBA")
        rgb.putalpha(alpha)

    return rgb


def _expand_bbox(bbox: tuple, image_size: tuple, padding_ratio: float) -> tuple:
    if padding_ratio <= 0:
        return bbox
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    pad_x = int(width * padding_ratio)
    pad_y = int(height * padding_ratio)
    new_x0 = max(0, x0 - pad_x)
    new_y0 = max(0, y0 - pad_y)
    new_x1 = min(image_size[0], x1 + pad_x)
    new_y1 = min(image_size[1], y1 + pad_y)
    return new_x0, new_y0, new_x1, new_y1


def preprocess_image(image: Image.Image, size: int = 256, input_resolution: int = 512,
                     padding_ratio: float = 0.1) -> Image.Image:
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
        image = image.crop(_expand_bbox(bbox, image.size, padding_ratio))

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


def match_color_statistics(target: Image.Image, reference: Image.Image) -> Image.Image:
    """Match color statistics of target to reference for consistency."""
    target_arr = np.asarray(target).astype(np.float32)
    ref_arr = np.asarray(reference).astype(np.float32)

    for channel in range(3):
        t = target_arr[..., channel]
        r = ref_arr[..., channel]
        t_mean, t_std = t.mean(), t.std()
        r_mean, r_std = r.mean(), r.std()
        if t_std < 1e-6:
            continue
        t = (t - t_mean) / t_std
        t = t * r_std + r_mean
        target_arr[..., channel] = t

    target_arr = np.clip(target_arr, 0, 255).astype(np.uint8)
    return Image.fromarray(target_arr)


def apply_postprocess(image: Image.Image,
                      reference: Image.Image,
                      detail_strength: float = 1.0,
                      color_strength: float = 1.0,
                      contrast_strength: float = 1.0,
                      sharpness_strength: float = 1.0,
                      gamma: float = 1.0,
                      match_colors: bool = False,
                      unsharp_radius: float = 1.2) -> Image.Image:
    """Enhance output with optional detail and color boosts."""
    output = image

    if match_colors:
        output = match_color_statistics(output, reference)

    if gamma != 1.0:
        arr = np.asarray(output).astype(np.float32) / 255.0
        arr = np.power(arr, gamma)
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        output = Image.fromarray(arr)

    if color_strength != 1.0:
        output = ImageEnhance.Color(output).enhance(color_strength)
    if contrast_strength != 1.0:
        output = ImageEnhance.Contrast(output).enhance(contrast_strength)
    if sharpness_strength != 1.0:
        output = ImageEnhance.Sharpness(output).enhance(sharpness_strength)

    if detail_strength != 1.0:
        percent = int(150 * detail_strength)
        output = output.filter(ImageFilter.UnsharpMask(radius=unsharp_radius, percent=percent, threshold=2))

    return output


def select_scheduler(pipe, scheduler_name: str):
    """Swap scheduler to try different synthesis behaviors."""
    from diffusers import DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler

    if scheduler_name == "Euler A (crisper edges)":
        return EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    if scheduler_name == "DPM++ 2M (detailed)":
        return DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    return DDIMScheduler.from_config(pipe.scheduler.config)


def generate_rotation_set(image: Image.Image, angle: float = 30,
                          num_steps: int = 75, guidance: float = 3.0,
                          num_frames: int = 3, black_white: bool = False,
                          input_resolution: int = 512,
                          padding_ratio: float = 0.1,
                          scheduler_name: str = "DDIM (balanced)",
                          detail_strength: float = 1.0,
                          color_strength: float = 1.0,
                          contrast_strength: float = 1.0,
                          sharpness_strength: float = 1.0,
                          gamma: float = 1.0,
                          match_colors: bool = False):
    """Generate frames across rotation range from +angle to -angle."""

    # Preprocess the image once (removes background)
    print(f"Preprocessing image (input res: {input_resolution}px)...")
    processed = preprocess_image(
        image,
        input_resolution=input_resolution,
        padding_ratio=padding_ratio
    )

    pipe = load_pipeline()
    pipe.scheduler = select_scheduler(pipe, scheduler_name)

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

        frame = apply_postprocess(
            frame,
            reference=processed,
            detail_strength=detail_strength,
            color_strength=color_strength,
            contrast_strength=contrast_strength,
            sharpness_strength=sharpness_strength,
            gamma=gamma,
            match_colors=match_colors
        )

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


def process_image(image, angle, num_steps, guidance, num_frames, black_white, boomerang, input_resolution,
                  padding_ratio, scheduler_name, detail_strength, color_strength, contrast_strength,
                  sharpness_strength, gamma, match_colors):
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
            input_resolution=int(input_resolution),
            padding_ratio=padding_ratio,
            scheduler_name=scheduler_name,
            detail_strength=detail_strength,
            color_strength=color_strength,
            contrast_strength=contrast_strength,
            sharpness_strength=sharpness_strength,
            gamma=gamma,
            match_colors=match_colors
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
                    padding_slider = gr.Slider(
                        minimum=0.0, maximum=0.3, value=0.1, step=0.05,
                        label="Subject Padding (more = wider crop)"
                    )
                    scheduler_dropdown = gr.Dropdown(
                        choices=[
                            "DDIM (balanced)",
                            "Euler A (crisper edges)",
                            "DPM++ 2M (detailed)",
                        ],
                        value="DDIM (balanced)",
                        label="Scheduler"
                    )

                    with gr.Accordion("Enhancements", open=False):
                        detail_slider = gr.Slider(
                            minimum=0.8, maximum=1.6, value=1.1, step=0.05,
                            label="Detail Boost"
                        )
                        color_slider = gr.Slider(
                            minimum=0.8, maximum=1.5, value=1.1, step=0.05,
                            label="Color Boost"
                        )
                        contrast_slider = gr.Slider(
                            minimum=0.8, maximum=1.3, value=1.05, step=0.05,
                            label="Contrast Boost"
                        )
                        sharpness_slider = gr.Slider(
                            minimum=0.8, maximum=1.6, value=1.1, step=0.05,
                            label="Sharpness"
                        )
                        gamma_slider = gr.Slider(
                            minimum=0.8, maximum=1.2, value=1.0, step=0.02,
                            label="Tone Gamma (lower = brighter)"
                        )
                        match_colors_checkbox = gr.Checkbox(
                            value=True,
                            label="Match Colors to Input"
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
            inputs=[
                input_image,
                angle_slider,
                steps_slider,
                guidance_slider,
                frames_slider,
                bw_checkbox,
                boomerang_checkbox,
                input_res_slider,
                padding_slider,
                scheduler_dropdown,
                detail_slider,
                color_slider,
                contrast_slider,
                sharpness_slider,
                gamma_slider,
                match_colors_checkbox,
            ],
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
    parser = argparse.ArgumentParser(description="Zero-1-to-3 Novel View Synthesis App")
    parser.add_argument("--skip-preload", action="store_true", help="Skip model preload for faster UI startup.")
    args = parser.parse_args()
    print("=" * 50)
    print("Zero-1-to-3 Novel View Synthesis")
    print("=" * 50)
    print(f"Device: {DEVICE}")
    print("Loading model (this may take a moment)...")
    print("=" * 50)

    # Pre-load the pipeline unless requested to skip
    if not args.skip_preload:
        load_pipeline()

    print("Starting web UI at http://127.0.0.1:7860")

    # Launch Gradio
    app = create_ui()
    app.launch(share=False, server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
