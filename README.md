# Basketball Box Score AI

Python pipeline that processes basketball game video to generate shot-by-shot statistics and player box scores. Uses computer vision (object detection, tracking, OCR) via Roboflow models and PyTorch.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/basketball-box-score-ai.git
cd basketball-box-score-ai

# Install dependencies (requires uv package manager)
uv sync
```

## Environment Setup

Create a `.env` file with:

```bash
ROBOFLOW_API_KEY=your_roboflow_api_key
HF_TOKEN=your_huggingface_token  # Optional, for HuggingFace model access
```

## CLI Commands

### Test Video Loading

```bash
uv run hoopstats test-video --video path/to/video.mp4
```

### Single Frame Detection

Detect all objects in a single frame using RF-DETR:

```bash
uv run hoopstats detect-frame --video path/to/video.mp4 --out data/outputs --frame 0
uv run hoopstats detect-frame --video path/to/video.mp4 --out data/outputs --filter players
```

### Full Video Detection

Run detection on every frame (no tracking, per-frame boxes):

```bash
uv run hoopstats detect-video --video path/to/video.mp4 --out data/outputs
uv run hoopstats detect-video --video path/to/video.mp4 --out data/outputs --filter players
```

### Full Video Tracking (SAM2)

Run SAM2 mask-based tracking across the full video:

```bash
uv run hoopstats track-video --video path/to/video.mp4 --out data/outputs
```

See [SAM2 Setup](#sam2-setup-for-track-video) below for installation requirements.

### Other Commands

```bash
# Test ByteTrack tracking on a short segment
uv run hoopstats test-tracking --video path/to/video.mp4 --out data/outputs --frames 30

# Test jersey number OCR
uv run hoopstats test-numbers --video path/to/video.mp4 --out data/outputs --frames 5

# Run full game processing pipeline
uv run hoopstats process-game --video path/to/video.mp4 --out data/outputs
```

## SAM2 Setup (for track-video)

The `track-video` command uses SAM2 (Segment Anything Model 2) for high-quality mask-based tracking. SAM2 requires separate installation:

### 1. Clone and Install SAM2

```bash
# Clone the SAM2 repository
git clone https://github.com/facebookresearch/sam2.git
cd sam2

# Install SAM2
pip install -e .
```

### 2. Download Checkpoints

Download the SAM2 checkpoints from the [official releases](https://github.com/facebookresearch/sam2#download-checkpoints):

```bash
cd sam2/checkpoints
# Download the large model (best quality)
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

Available models (quality vs speed tradeoff):
- `sam2.1_hiera_large.pt` - Highest quality masks (recommended)
- `sam2.1_hiera_base_plus.pt` - Good balance
- `sam2.1_hiera_small.pt` - Faster, lower quality
- `sam2.1_hiera_tiny.pt` - Fastest, lowest quality

### 3. Configure Paths

Either set environment variables in `.env`:

```bash
SAM2_CHECKPOINT=/path/to/sam2/checkpoints/sam2.1_hiera_large.pt
SAM2_CONFIG=/path/to/sam2/configs/sam2.1/sam2.1_hiera_l.yaml
```

Or pass them as CLI arguments:

```bash
uv run hoopstats track-video \
  --video path/to/video.mp4 \
  --out data/outputs \
  --sam2-checkpoint /path/to/sam2/checkpoints/sam2.1_hiera_large.pt \
  --sam2-config /path/to/sam2/configs/sam2.1/sam2.1_hiera_l.yaml
```

### Config Files by Model Size

| Checkpoint | Config File |
|------------|-------------|
| `sam2.1_hiera_large.pt` | `configs/sam2.1/sam2.1_hiera_l.yaml` |
| `sam2.1_hiera_base_plus.pt` | `configs/sam2.1/sam2.1_hiera_b+.yaml` |
| `sam2.1_hiera_small.pt` | `configs/sam2.1/sam2.1_hiera_s.yaml` |
| `sam2.1_hiera_tiny.pt` | `configs/sam2.1/sam2.1_hiera_t.yaml` |

## Output Structure

```
data/outputs/
  single_frame_detection/
    detect_frame_0.jpg
  video_detection/
    video-detection.mp4
  video_tracking/
    video-mask.mp4          # SAM2 mask tracking output
```

## Architecture

The pipeline processes video through these stages:

1. **Video Loading** (`video_io.py`) - Frame extraction via OpenCV
2. **Detection** (`detection.py`) - RF-DETR model detects players, ball, rim, jersey numbers
3. **Tracking** - Either ByteTrack (`tracking.py`) or SAM2 (`sam2_tracking.py`)
4. **Team Assignment** (`teams.py`) - Color-based clustering into home/away teams
5. **Jersey OCR** (`numbers.py`) - Reads jersey numbers, validates across frames
6. **Homography** (`homography.py`) - Court keypoint detection maps pixels to court coordinates
7. **Events** (`events.py`) - Shot detection and make/miss classification
8. **Stats Export** (`stats.py`) - Generates `shots.csv` and `box_score.csv`

## Detection Classes

The player detection model outputs:
- 0: ball
- 1: ball-in-basket
- 2: number (jersey)
- 3: player
- 4: player-in-possession
- 5: player-jump-shot
- 6: player-layup-dunk
- 7: player-shot-block
- 8: referee
- 9: rim

## License

See [LICENSE](LICENSE) for details.
