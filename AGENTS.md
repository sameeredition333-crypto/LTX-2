# AGENTS.md

## Project Overview

**LTX-2** by Lightricks is a DiT-based audio-video generation model (22B params). This repo contains the Python implementation: `ltx-core` (model architecture), `ltx-pipelines` (inference pipelines), `ltx-kernels` (optional CUDA kernels), and `ltx-trainer` (fine-tuning toolkit).

**This is NOT a web application.** It is a CLI/Python library for GPU-based video generation. There is no built-in web server or frontend.

## Running the Model

Requires an **NVIDIA GPU with CUDA** and ~66 GiB of model weights from Hugging Face (`Lightricks/LTX-2.5`, gated repo).

```bash
uv sync --extra natten
hf download Lightricks/LTX-2.5 <files...> --local-dir models/ltx-2.5
uv run python -m ltx_pipelines.distilled --transformer-path ... --output-path output.mp4 --prompt "..."
```

## Web UI (Gradio)

A Gradio web interface was added at `webapp.py`. It wraps the `DistilledPipeline` with lazy imports — the UI starts on any machine, but generation requires CUDA + model weights.

```bash
docker compose -f docker-compose.base44.yml up -d   # UI on port 3000
# Or directly: python webapp.py --host 0.0.0.0 --port 3000
```

On a GPU machine, install full deps (`uv sync --extra natten`) so the pipeline can be imported at generation time.

## Key Architecture

- **`DistilledPipeline`** (`packages/ltx-pipelines/src/ltx_pipelines/distilled.py`): Two-stage pipeline — stage 1 generates at half resolution, stage 2 upsamples 2x and refines.
- **`ModelPaths`** (`packages/ltx-pipelines/src/ltx_pipelines/utils/model_paths.py`): Normalized paths for monolith or split checkpoint layouts. The web UI uses split layout.
- **`encode_video`** (`packages/ltx-pipelines/src/ltx_pipelines/utils/media_io/encode.py`): Encodes decoded frames + audio to MP4 (H.264).
- **`OffloadMode`**: `none` (all on GPU), `cpu` (stream from RAM), `disk` (stream from disk).
- **`QuantizationPolicy`**: `fp8-cast` (bf16→fp8 on the fly), `fp8-scaled-mm` (Hopper+ native FP8).

## Sandbox Limitations

This sandbox has **no GPU** and **20 GB disk** — the model (66 GiB weights) cannot run here. The Gradio UI starts and is fully interactive, but clicking "Generate" will show a clear error explaining what's needed.
