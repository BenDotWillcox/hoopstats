# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Basketball game video analysis pipeline that processes broadcast footage to generate shot-by-shot statistics and player box scores. Uses computer vision (object detection, tracking, OCR) via Roboflow models and PyTorch.

## Build & Run Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Run CLI commands
uv run hoopstats test-video --video path/to/video.mp4
uv run hoopstats detect-frame --video path/to/video.mp4 --out data/outputs --frame 0
uv run hoopstats detect-video --video path/to/video.mp4 --out data/outputs
uv run hoopstats track-video --video path/to/video.mp4 --out data/outputs  # SAM2 mask tracking
uv run hoopstats test-tracking --video path/to/video.mp4 --out data/outputs --frames 30
uv run hoopstats test-numbers --video path/to/video.mp4 --out data/outputs --frames 5
uv run hoopstats process-game --video path/to/video.mp4 --out data/outputs
```

## Environment Setup

Requires a `.env` file with:
- `ROBOFLOW_API_KEY` - Required for all detection/OCR models
- `HF_TOKEN` - For HuggingFace model access
- `SAM2_CHECKPOINT` - Path to SAM2 checkpoint (optional, for track-video command)
- `SAM2_CONFIG` - Path to SAM2 config yaml (optional, for track-video command)

For SAM2 setup, see README.md.

## Architecture

The pipeline processes video through these sequential stages:

1. **Video Loading** (`video_io.py`) - Frame extraction via OpenCV
2. **Detection** (`detection.py`) - Roboflow model detects players, ball, rim, jersey numbers
3. **Tracking** - ByteTrack (`tracking.py`) or SAM2 mask tracking (`sam2_tracking.py`)
4. **Team Assignment** (`teams.py`) - Color-based clustering into home/away teams
5. **Jersey OCR** (`numbers.py`) - Reads jersey numbers, validates across frames
6. **Homography** (`homography.py`) - Court keypoint detection maps pixels to court coordinates (feet)
7. **Events** (`events.py`) - Shot detection and make/miss classification
8. **Stats Export** (`stats.py`) - Generates `shots.csv` and `box_score.csv`

`GameProcessor` in `pipeline.py` orchestrates all stages.

## Key Dependencies

- `inference` - Roboflow model inference
- `supervision` - Detection/tracking utilities (ByteTrack, annotations)
- `sports` - Roboflow's basketball court config, team classifier, view transformer
- `paddleocr` - Jersey number OCR
- `sam2` - SAM2 for mask-based video tracking (optional, separate install)

## Detection Classes

The player detection model outputs these classes:
- 0: ball, 1: ball-in-basket, 2: number (jersey), 3: player, 4: player-in-possession
- 5: player-jump-shot, 6: player-layup-dunk, 7: player-shot-block, 8: referee, 9: rim

Player classes for tracking: `{3, 4, 5, 6, 7}` (defined in `teams.py`)

## Data Model

`ShotEvent` (in `models.py`) is the core output: period, game clock, shooter info, result (make/miss), shot type (2PT/3PT/FT), court coordinates.
