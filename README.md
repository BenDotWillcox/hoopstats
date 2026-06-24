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

# Run full game processing pipeline (shots.csv + box_score.csv)
uv run hoopstats process-game --video path/to/video.mp4 --out data/outputs
uv run hoopstats process-game --video path/to/video.mp4 --out data/outputs --max-frames 300 --no-number-ocr
```

### Reproducible Demo

Runs every pipeline stage on one sample clip and writes a `report.md` plus all
artifacts (annotated frames, court-map video, trajectories, `stats/shots.csv`,
`stats/box_score.csv`):

```bash
uv run hoopstats demo --video data/raw/clip.mp4 --out data/demo
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

## Shot Detection & Box Score

The shot pipeline (`events.py` → `pipeline.py` → `stats.py`) is deliberately
heuristic: every signal comes from the detection model's action classes plus
court homography, with no extra learned components.

### How it works

1. **Attempt detection** — frames containing a `player-jump-shot` or
   `player-layup-dunk` detection are grouped into one attempt window
   (gap tolerance ~0.7s). The pose must persist for ≥2 frames and reach
   confidence ≥0.5 once, which filters single-frame false positives. The
   *release frame* is the last pose frame.
2. **Make/miss** — an attempt is a make if a `ball-in-basket` detection group
   starts within 3s of the release. Each `ball-in-basket` group credits at
   most one attempt (nearest preceding).
3. **Shooter attribution** — the ByteTrack player track with the highest IoU
   against the shot-pose box (release frame ±5) is the shooter. Team comes
   from the appearance-based team classifier (majority vote per track);
   jersey number from OCR on `number` detections inside the shooter's box
   near the release frame, when one is visible.
4. **Shot location** — the shooter's feet (bottom-center of the box) are
   projected through the court homography at the release frame (nearest
   frame within 1s if keypoints fail there). Distance to the nearest hoop
   classifies 2PT vs 3PT using the NBA arc (23.75 ft, 22 ft in the corners).
5. **Export** — `shots.csv` is the shot log; `box_score.csv` aggregates
   FGM/FGA/FG%/3PM/3PA/PTS per `team#number`, with unidentified shooters
   bucketed per team so team totals still reconcile.

### Model assumptions

- Broadcast camera angle similar to the detection model's training data; the
  action classes (`player-jump-shot`, `ball-in-basket`) carry all event
  semantics.
- The shooter is on the ground at release (feet projection is only valid for
  the bottom of the bounding box; a player mid-jump projects slightly long).
- One period / continuous game clock per processed segment; clock is linear
  interpolation, not scoreboard OCR.
- Free throws are not separately detected (no FT classification yet).

### Known failure modes

- **Missed shot poses**: fadeaways, tip-ins, and putbacks often never trigger
  the jump-shot/layup classes → missed attempts (false negatives dominate).
- **Pump fakes** held long enough can register as attempts (false positives).
- **`ball-in-basket` flicker or occlusion** near the rim → makes scored as
  misses; a made shot followed quickly by another attempt can mis-assign the
  make.
- **Tracking ID switches** during crowded paint play → wrong shooter, wrong
  team attribution.
- **Homography dropout** when too few court keypoints are visible (zoomed-in
  or transition shots) → shot logged without coordinates, defaults to 2PT.
- **Jersey OCR** requires the number to face the camera near the release;
  most shots will have no number and fall into the team's `unknown` bucket.
- **Broadcast cuts** (replays, alternate angles, overlays) confuse detection
  and homography → handle with a segments file (see below), not by cutting.
- **Free throws** are not detected and would be miscounted as field goals if
  not excluded → tag them `ft` in ground truth (see below).
- **Per-segment team ids** are independent across segments (the appearance
  classifier is fit per range), so the full-game `box_score.csv` does not align
  team 0/1 across segments; evaluation handles this with permutation-invariant
  team scoring.

### Metrics: annotate ground truth, then evaluate

Label shot events with the built-in annotator (OpenCV window + keyboard, no
models or API needed):

```bash
uv run hoopstats annotate --video data/raw/game.mp4 --out data/labels/game_ground_truth.csv
```

Controls: `space` play/pause, `a`/`d` step a frame, `s`/`w` jump 1s,
`m` mark make, `x` mark miss (then answer team/jersey/zone prompts in the
terminal), `u` undo, `q` save and quit. Labels are saved after every mark and
reloaded on restart, so sessions are resumable. Location is labeled as a
coarse zone (`paint`, `midrange`, `corner3`, `arc3`); mark free throws with
zone `ft`.

Then score the pipeline's predictions against the labels:

```bash
uv run hoopstats process-game --video data/raw/game.mp4 --out data/outputs/game
uv run hoopstats evaluate --shots data/outputs/game/shots.csv \
  --labels data/labels/game_ground_truth.csv --fps 30 --report-out data/outputs/game/eval.md
```

The report covers:

- **Attempt detection** (field goals): precision/recall/F1, matching
  predictions to labels within ±2s (greedy, closest-first).
- **Make/miss**: accuracy on matched attempts.
- **Attribution**: team and jersey-number accuracy on matched attempts.
  Team accuracy is permutation-invariant (the classifier's 0/1 are arbitrary
  and only defined up to a swap) and is reported only when both teams appear.
  Blank labels ("couldn't tell") are excluded, not penalized.
- **Shot type**: predicted 2PT/3PT vs the labeled zone.
- **Free throws**: counted separately and excluded from FG metrics (see below).
- Frame lists of false positives and missed attempts for debugging.

### Evaluating uncut broadcasts (segments + free throws)

A full uncut broadcast has replays, alternate camera angles, on-screen
overlays, and dead time the system was never meant to handle — and is too long
to hold in memory. Both problems are solved with a **segments file** that
lists the in-scope (main tactical camera, live play) frame ranges in the
*original* video:

```
start_frame,end_frame,type
0,1830,live
1830,2400,replay
2400,5200,live
```

Mark these ranges frame-accurately with the interactive tool (same controls as
`annotate`, plus `i` for the in-point and `o` for the out-point of each
live-play stretch):

```bash
uv run hoopstats mark-segments --video data/raw/game.mp4 --out data/labels/game_segments.csv
```

Pad each stretch ~1–2s past the last shot before a whistle so the
`ball-in-basket` frames that decide make/miss are not clipped. Or hand-write
the CSV — only the `start_frame,end_frame` columns are required (rows with no
`type` column are all treated as in scope).

Pass it to both commands. During processing it bounds memory (only one segment
is loaded at a time) and skips out-of-scope footage; during evaluation it
filters predictions and labels to the same ranges, so metrics reflect only the
footage the system targets:

```bash
uv run hoopstats process-game --video data/raw/game.mp4 --out data/outputs/game \
  --segments data/labels/game_segments.csv
uv run hoopstats evaluate --shots data/outputs/game/shots.csv \
  --labels data/labels/game_ground_truth.csv --segments data/labels/game_segments.csv \
  --fps 30 --report-out data/outputs/game/eval.md
```

Frame indices stay absolute throughout, so labels annotated against the full
video line up without remapping. (To process a single arbitrary window instead
of a segments file, use `--start-frame` / `--end-frame`.)

**Free throws** are out of scope for v1: the detection model has no FT class,
and a free throw is geometrically near-identical to a center midrange jumper —
distinguishing them needs game-clock/lane context the pipeline doesn't model.
Mark free throws with zone `ft` in your ground truth; evaluation pulls them out
of the FG precision/recall and reports them separately, including a count of
predictions that fired on a free throw (which would otherwise inflate FGA in
the box score). Do **not** cut the video to remove replays — cutting renumbers
frames and invalidates your annotation; the segments file is the supported path.

The heuristic layer (`events.py`, `stats.py`, `evaluate.py`, `segments.py`) is
covered by unit tests over synthetic detections:

```bash
uv run pytest
```

## Portfolio Report

`hoopstats report` assembles the pipeline's stage artifacts (annotated frames,
top-down court map), the shot log, the box score, and the evaluation metrics
into a single self-contained `pipeline.html` — a vertical-slice walkthrough that
shows each stage working on one game. Images are base64-embedded, so the file
can be opened directly, hosted, or screenshotted. Re-run it to refresh as the
pipeline improves.

```bash
uv run hoopstats report \
  --frames-dir data/outputs \
  --court-video data/outputs/court_map_video/bos-nyk-court-map.mp4 \
  --debug-video data/outputs/court_map_video/bos-nyk-tracking-debug.mp4 \
  --stats-dir data/outputs/game \
  --out data/portfolio/pipeline.html
```

`--frames-dir` is searched recursively for the stage images
(`detect_frame_*`, `keypoints_frame_*`, `teams_frame_*`, `court_frame_*` — as
produced by `detect-frame`, `detect-keypoints`, `map-court`, or `demo`).
`--court-video` / `--debug-video` feature short clips at the top ("See it
track"); they're copied next to the HTML into an `assets/` folder and
referenced by relative path, so move the two together. `--stats-dir` supplies
`shots.csv` / `box_score.csv` for the in-progress extension panels. Accuracy
metrics are off by default (tracking-first framing); add `--metrics --labels
ground_truth.csv [--segments segs.csv]` to show them. Stages without an
artifact are shown as "not generated", so the report degrades gracefully.

The report leads with the tracking pipeline and frames the shot/box-score work
as an in-progress extension.

### Tracking smoothing

`map-court-video` temporally smooths the court mapping by default:

- **Homography**: EMA on the per-frame 3x3 matrix, and on frames whose court
  keypoints momentarily fail it reuses the last good homography (gap-fill) so
  players don't flicker out.
- **Player paths**: after collecting per-track court positions, the whole clip
  is cleaned with the `sports` library's `clean_paths` — it removes
  teleport-like outliers (from bad single-frame homographies), interpolates
  short gaps, and Savitzky-Golay smooths each player's trajectory. This is the
  same routine the reference notebook uses and is much stronger than a causal
  EMA (non-causal, with outlier rejection and gap-filling).
- **Ball**: a light EMA (the ball is sparse and fast).

Disable all of this with `--no-smooth` to compare.

### SAM2 tracking (GPU, notebook-quality)

ByteTrack still churns tracks on CPU. For the smoothest result, `map-court-video`
has a `--tracker sam2` path that prompts the player set on frame 0 and propagates
masks with SAM2, so the same players are tracked every frame with stable ids
(then `clean_paths` runs as above). It needs a CUDA GPU and the SAM2 real-time
build — easiest on Colab:

```bash
uv run hoopstats map-court-video --video clip.mp4 --out data/outputs --debug \
  --tracker sam2 --sam2-checkpoint <ckpt>.pt --sam2-config configs/sam2.1/sam2.1_hiera_l.yaml
```

SAM2 prompts on frame 0, so it's for a **short clip** (one possession, same
players on screen) — not a full game with substitutions/cuts. See
[`notebooks/hoopstats_colab.ipynb`](notebooks/hoopstats_colab.ipynb) for a
ready-to-run Colab that installs SAM2, runs the SAM2 court map, and builds the
report on a GPU. Whole-game stats stay on CPU ByteTrack via `process-game`.

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
