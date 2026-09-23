"""Gradio web UI for the LTX-2 DistilledPipeline.

Run on a GPU machine with model weights downloaded:
    uv run python webapp.py --models-dir models/ltx-2.5

The UI starts even without a GPU (pipeline imports are deferred to generation
time), so you can explore the interface anywhere.  Generation requires CUDA.
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path

import gradio as gr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults — match the README's `models/ltx-2.5` folder layout.
# ---------------------------------------------------------------------------
DEFAULT_MODELS_DIR = "models/ltx-2.5"
DEFAULT_HEIGHT = 720
DEFAULT_WIDTH = 1280
DEFAULT_NUM_FRAMES = 121
DEFAULT_FRAME_RATE = 30
DEFAULT_SEED = 42

# Resolutions that look good and are divisible by 64 (stage-2 requirement).
RESOLUTION_CHOICES = [
    "768 x 1344",
    "720 x 1280",
    "544 x 960",
    "512 x 896",
    "480 x 832",
]

QUANTIZATION_CHOICES = ["none", "fp8-cast", "fp8-scaled-mm"]
OFFLOAD_CHOICES = ["none", "cpu", "disk"]


def _split_paths(models_dir: str) -> dict[str, str]:
    """Build the default split-pack paths from a models directory."""
    base = Path(models_dir)
    return {
        "transformer": str(base / "diffusion_models" / "ltx-2.5-22b-distilled-transformer-bf16.safetensors"),
        "text_encoder": str(base / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"),
        "video_vae": str(base / "vae" / "ltx-2.5-video-vae-bf16.safetensors"),
        "audio_vae": str(base / "vae" / "ltx-2.5-audio-vae-bf16.safetensors"),
        "spatial_upsampler": str(base / "latent_upscale_models" / "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"),
    }


def generate(
    prompt: str,
    models_dir: str,
    resolution: str,
    num_frames: int,
    frame_rate: float,
    seed: int,
    quantization: str,
    offload: str,
    enhance_prompt: bool,
    image: str | None,
    image_strength: float,
) -> tuple[str | None, str]:
    """Run the DistilledPipeline and return (video_path, status_message)."""
    if not prompt.strip():
        return None, "⚠️ Please enter a prompt."

    # Lazy import — torch / ltx_core / ltx_pipelines are only needed for generation.
    try:
        import torch  # noqa: F811
        from ltx_pipelines.distilled import DistilledPipeline
        from ltx_pipelines.utils.media_io import encode_video
        from ltx_pipelines.utils.model_paths import ModelPaths
        from ltx_pipelines.utils.types import OffloadMode
        from ltx_pipelines.utils.args import ImageConditioningInput
        from ltx_pipelines.utils.helpers import get_device
        from ltx_pipelines.utils.constants import AUTO_TILING
        from ltx_core.quantization import QuantizationPolicy
        from ltx_pipelines.utils.quantization_factory import QuantizationKind
    except ImportError as exc:
        return None, (
            f"❌ Cannot import pipeline dependencies: {exc}\n\n"
            "Make sure you are running on a GPU machine with the project installed:\n"
            "  uv sync --extra natten\n\n"
            "And model weights downloaded to the models directory."
        )

    if not torch.cuda.is_available():
        return None, (
            "❌ No CUDA GPU detected. LTX-2 requires an NVIDIA GPU to generate video.\n\n"
            "Run this app on a machine with an NVIDIA GPU and CUDA installed."
        )

    paths = _split_paths(models_dir)

    # Verify model files exist.
    missing = [name for name, p in paths.items() if not Path(p).exists()]
    if missing:
        return None, (
            f"❌ Missing model files: {', '.join(missing)}\n\n"
            f"Expected under: {models_dir}\n"
            "Download weights from Hugging Face:\n"
            "  hf download Lightricks/LTX-2.5 \\\n"
            "    diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors \\\n"
            "    text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \\\n"
            "    vae/ltx-2.5-video-vae-bf16.safetensors \\\n"
            "    vae/ltx-2.5-audio-vae-bf16.safetensors \\\n"
            "    latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \\\n"
            "    --local-dir models/ltx-2.5"
        )

    # Parse resolution.
    h, w = (int(x.strip()) for x in resolution.split("x"))

    # Build ModelPaths (split layout).
    model_paths = ModelPaths.from_split(
        transformer_path=paths["transformer"],
        text_encoder_path=paths["text_encoder"],
        video_vae_path=paths["video_vae"],
        audio_vae_path=paths["audio_vae"],
    )

    # Quantization.
    quant_policy = None
    if quantization != "none":
        quant_policy = QuantizationKind(quantization).to_policy(checkpoint_path=paths["transformer"])

    # Offload.
    offload_mode = OffloadMode(offload)

    # Image conditioning.
    images = []
    if image is not None:
        images.append(ImageConditioningInput(path=image, frame_idx=0, strength=image_strength))

    device = get_device()

    try:
        logger.info("Loading pipeline (this may take a while)...")
        pipeline = DistilledPipeline(
            model_paths=model_paths,
            spatial_upsampler_path=paths["spatial_upsampler"],
            loras=(),
            device=device,
            quantization=quant_policy,
            offload_mode=offload_mode,
        )

        logger.info("Generating video...")
        result = pipeline(
            prompt=prompt,
            seed=seed,
            height=h,
            width=w,
            frame_rate=frame_rate,
            images=images,
            num_frames=num_frames,
            enhance_prompt=enhance_prompt,
            tiling_config=AUTO_TILING,
        )

        output_path = str(Path(tempfile.gettempdir()) / "ltx2_output.mp4")
        from ltx_pipelines.utils.media_io import get_video_chunks_number  # type: ignore  # noqa: PLC0415

        encode_video(
            video=result.video,
            fps=int(frame_rate),
            audio=result.audio,
            output_path=output_path,
            video_chunks_number=get_video_chunks_number(result.num_frames, result.tiling_config),
        )

        return output_path, f"✅ Video generated successfully ({num_frames} frames @ {frame_rate} fps, {w}x{h})."

    except Exception as exc:
        logger.exception("Generation failed")
        return None, f"❌ Generation failed: {exc}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui(models_dir: str) -> gr.Blocks:
    default_paths = _split_paths(models_dir)

    with gr.Blocks(
        title="LTX-2 Video Generator",
        theme=gr.themes.Soft(primary_hue="amber", secondary_hue="orange"),
        css="""
        #generate-btn { font-size: 1.1rem; font-weight: bold; }
        .status-box { min-height: 60px; }
        """,
    ) as app:
        gr.Markdown(
            "# 🎬 LTX-2 Video Generator\n"
            "Generate synchronized audio-video from text prompts using the LTX-2.5 distilled pipeline.\n\n"
            "**Requires an NVIDIA GPU and downloaded model weights.**"
        )

        with gr.Row():
            # ---- Left column: inputs ----
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe the video you want to generate...",
                    lines=4,
                    value=(
                        "A medium close-up shot features a Caucasian man with a beard, wearing a "
                        "green and white baseball cap, looking directly at the camera with deep "
                        "concentration. The camera remains static with a shallow depth of field."
                    ),
                )

                image_input = gr.Image(
                    label="Reference Image (optional, for image-to-video)",
                    type="filepath",
                )

                generate_btn = gr.Button("🎬 Generate Video", variant="primary", elem_id="generate-btn")

                status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    elem_classes="status-box",
                    value="Ready. Enter a prompt and click Generate.",
                )

            # ---- Right column: output + settings ----
            with gr.Column(scale=1):
                video_output = gr.Video(label="Generated Video")

        with gr.Accordion("⚙️ Advanced Settings", open=False):
            with gr.Row():
                resolution = gr.Dropdown(
                    label="Resolution",
                    choices=RESOLUTION_CHOICES,
                    value=f"{DEFAULT_HEIGHT} x {DEFAULT_WIDTH}",
                )
                num_frames = gr.Slider(
                    label="Number of Frames",
                    minimum=9,
                    maximum=257,
                    step=8,
                    value=DEFAULT_NUM_FRAMES,
                    info="Must be 8k+1 (9, 17, 25, ...). More frames = longer video.",
                )
                frame_rate = gr.Slider(
                    label="Frame Rate (fps)",
                    minimum=1,
                    maximum=60,
                    step=1,
                    value=DEFAULT_FRAME_RATE,
                )
                seed = gr.Number(label="Seed", value=DEFAULT_SEED, precision=0)

            with gr.Row():
                quantization = gr.Dropdown(
                    label="Quantization",
                    choices=QUANTIZATION_CHOICES,
                    value="none",
                    info="fp8-cast for lower VRAM (bf16 checkpoints). fp8-scaled-mm on Hopper+.",
                )
                offload = gr.Dropdown(
                    label="Weight Offload",
                    choices=OFFLOAD_CHOICES,
                    value="none",
                    info="cpu: stream from RAM. disk: stream from disk (lowest VRAM).",
                )
                image_strength = gr.Slider(
                    label="Image Conditioning Strength",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.8,
                    info="How strongly the reference image guides the first frame.",
                )
                enhance_prompt = gr.Checkbox(
                    label="Enhance Prompt",
                    value=False,
                    info="Use the model's prompt enhancer for more descriptive prompts.",
                )

        with gr.Accordion("📁 Model Paths", open=False):
            models_dir_input = gr.Textbox(
                label="Models Directory",
                value=models_dir,
                info="Root directory containing the downloaded LTX-2.5 weights.",
            )
            gr.Markdown(
                "Expected layout under the models directory:\n"
                "```\n"
                f"{default_paths['transformer']}\n"
                f"{default_paths['text_encoder']}\n"
                f"{default_paths['video_vae']}\n"
                f"{default_paths['audio_vae']}\n"
                f"{default_paths['spatial_upsampler']}\n"
                "```"
            )

        generate_btn.click(
            fn=generate,
            inputs=[
                prompt,
                models_dir_input,
                resolution,
                num_frames,
                frame_rate,
                seed,
                quantization,
                offload,
                enhance_prompt,
                image_input,
                image_strength,
            ],
            outputs=[video_output, status],
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="LTX-2 Gradio Web UI")
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR, help="Directory with downloaded model weights.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=3000, help="Bind port.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    app = build_ui(args.models_dir)
    app.launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
