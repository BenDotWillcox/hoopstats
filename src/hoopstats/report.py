"""
Portfolio report generator.

Assembles the pipeline's stage artifacts (annotated frames + top-down court map)
plus the shot log, box score, and evaluation metrics into a single
self-contained `pipeline.html` — a "vertical slice" walkthrough that shows each
stage of the broadcast-to-box-score pipeline working on one game.

Self-contained: images are base64-embedded, so the file can be opened directly,
hosted, or screenshotted for a portfolio. Re-run as the project improves to
refresh it.
"""

import base64
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class Stage:
    key: str
    title: str
    model: str
    caption: str
    image_glob: str


# Pipeline stages in order, each tied to an artifact the pipeline already emits.
STAGES: List[Stage] = [
    Stage(
        "detection", "1. Object detection",
        "RF-DETR · roboflow basketball-player-detection",
        "Every frame is run through the detector: players, ball, rim, referees, "
        "and jersey-number regions, each with a class and confidence.",
        "detect_frame_*.jpg",
    ),
    Stage(
        "keypoints", "2. Court keypoints",
        "roboflow basketball-court-detection",
        "Court landmarks (pink) are detected and used to fit a homography that "
        "maps image pixels to real court coordinates in feet.",
        "keypoints_frame_*.jpg",
    ),
    Stage(
        "teams", "3. Tracking & team assignment",
        "ByteTrack + appearance clustering",
        "Players are tracked across frames and clustered into two teams by "
        "jersey appearance; boxes are recoloured by team.",
        "teams_frame_*.jpg",
    ),
    Stage(
        "court", "4. Court mapping",
        "homography projection",
        "Each player's foot position is projected onto a top-down court, giving "
        "real (x, y) positions used for shot locations and 2PT/3PT calls.",
        "court_frame_*.jpg",
    ),
]


def _data_uri(path: Path) -> Optional[str]:
    if path is None or not path.exists():
        return None
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _find_image(frames_dir: Path, pattern: str) -> Optional[Path]:
    """Most-recent file matching pattern anywhere under frames_dir."""
    matches = sorted(frames_dir.rglob(pattern), key=lambda p: p.stat().st_mtime,
                     reverse=True)
    return matches[0] if matches else None


def _read_csv(path: Optional[Path]) -> List[dict]:
    if path is None or not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _pct(v) -> str:
    return f"{v:.0%}" if isinstance(v, (int, float)) else "—"


def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def generate_pipeline_html(
    out_path: Path,
    frames_dir: Path,
    shots: List[dict],
    box_score: List[dict],
    metrics: Optional[dict],
    show_metrics: bool = False,
    videos: Optional[List[dict]] = None,
    title: str = "HoopStats — Basketball Player Tracking",
    subtitle: str = "Computer-vision pipeline that detects, tracks, and maps every "
                    "player from broadcast footage onto a real court in feet. An "
                    "experimental shot-detection & box-score layer is in progress.",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Hero videos — copied next to the HTML and referenced by relative path
    # (MP4 is too large to base64-embed; the report ships as HTML + assets/).
    hero_html = ""
    if videos:
        assets_dir = out_path.parent / "assets"
        blocks = []
        for v in videos:
            src = Path(v["path"])
            if not src.exists():
                continue
            assets_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, assets_dir / src.name)
            blocks.append(f"""
            <figure class="vid">
              <video src="assets/{_esc(src.name)}" autoplay muted loop playsinline controls></video>
              <figcaption>{_esc(v.get('caption', ''))}</figcaption>
            </figure>""")
        if blocks:
            hero_html = f"""
        <h2>See it track</h2>
        <div class="videos">{''.join(blocks)}</div>"""

    # Stage cards
    stage_html = []
    for st in STAGES:
        uri = _data_uri(_find_image(frames_dir, st.image_glob))
        img = (f'<img src="{uri}" alt="{_esc(st.title)}"/>' if uri else
               '<div class="missing">artifact not generated</div>')
        stage_html.append(f"""
        <section class="stage">
          <div class="stage-img">{img}</div>
          <div class="stage-text">
            <h3>{_esc(st.title)}</h3>
            <div class="model">{_esc(st.model)}</div>
            <p>{_esc(st.caption)}</p>
          </div>
        </section>""")

    # Metrics strip (opt-in; off by default for the tracking-first framing)
    metrics_html = ""
    if show_metrics and metrics:
        cards = [
            ("Attempt F1", _pct(metrics.get("f1")),
             f"P {_pct(metrics.get('precision'))} · R {_pct(metrics.get('recall'))}"),
            ("Make / miss", _pct(metrics.get("result_accuracy")),
             f"{metrics.get('result_correct', 0)}/{metrics.get('result_n', 0)} matched"),
            ("Shot type 2/3", _pct(metrics.get("shot_type_accuracy")),
             f"{metrics.get('shot_type_correct', 0)}/{metrics.get('shot_type_n', 0)} zoned"),
            ("Team", _pct(metrics.get("team_accuracy")),
             "permutation-invariant"),
        ]
        metric_cards = "".join(
            f'<div class="metric"><div class="m-val">{v}</div>'
            f'<div class="m-key">{_esc(k)}</div><div class="m-sub">{_esc(s)}</div></div>'
            for k, v, s in cards)
        n_label = metrics.get('n_label', 0)
        metrics_html = f"""
        <div class="metrics">
          <h3>Accuracy so far <span class="muted">· experimental · vs {n_label} hand-labeled field-goal attempts</span></h3>
          <div class="metric-row">{metric_cards}</div>
        </div>"""

    # Box score (top scorers, identified players only)
    box_html = ""
    if box_score:
        ranked = sorted(
            (r for r in box_score if not r["player"].endswith("#unknown")),
            key=lambda r: int(r["pts"]), reverse=True)[:10]
        rows = "".join(
            f"<tr><td>{_esc(r['player'])}</td><td>{r['team']}</td>"
            f"<td>{r['fgm']}/{r['fga']}</td><td>{r['tpm']}/{r['tpa']}</td>"
            f"<td>{r['pts']}</td></tr>" for r in ranked)
        box_html = f"""
        <div class="panel">
          <h3>Box score <span class="muted">· top scorers (team#jersey)</span></h3>
          <table><thead><tr><th>Player</th><th>Tm</th><th>FG</th><th>3PT</th>
          <th>PTS</th></tr></thead><tbody>{rows}</tbody></table>
        </div>"""

    # Shot log preview
    shots_html = ""
    if shots:
        preview = shots[:8]
        rows = "".join(
            f"<tr><td>{r['video_frame_idx']}</td><td>{r['result']}</td>"
            f"<td>{r['shot_type']}</td><td>{r['offense_team_id'] or '—'}"
            f"#{r['shooter_number'] or '?'}</td>"
            f"<td>{r['distance_ft'] or '—'}</td></tr>" for r in preview)
        shots_html = f"""
        <div class="panel">
          <h3>Shot log <span class="muted">· {len(shots)} detected shots → shots.csv</span></h3>
          <table><thead><tr><th>Frame</th><th>Result</th><th>Type</th>
          <th>Shooter</th><th>Dist ft</th></tr></thead><tbody>{rows}</tbody></table>
        </div>"""

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>
  :root {{ --bg:#0f1216; --card:#171c22; --line:#262d36; --fg:#e8edf2;
    --muted:#8a97a6; --accent:#36d07f; --accent2:#ff5d5d; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
    font:15px/1.55 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }}
  .wrap {{ max-width:980px; margin:0 auto; padding:40px 24px 64px; }}
  header h1 {{ margin:0 0 6px; font-size:30px; letter-spacing:-.02em; }}
  header p {{ margin:0; color:var(--muted); max-width:680px; }}
  .chips {{ margin:16px 0 8px; }}
  .chip {{ display:inline-block; background:var(--card); border:1px solid var(--line);
    color:var(--muted); border-radius:999px; padding:3px 11px; font-size:12px; margin:0 6px 6px 0; }}
  h2 {{ font-size:18px; margin:36px 0 14px; }}
  .muted {{ color:var(--muted); font-weight:400; font-size:13px; }}
  .intro {{ color:var(--muted); max-width:700px; margin:0 0 16px; }}
  .videos {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:16px; }}
  .vid {{ margin:0; background:var(--card); border:1px solid var(--line); border-radius:12px; padding:10px; }}
  .vid video {{ width:100%; border-radius:8px; display:block; background:#000; }}
  .vid figcaption {{ color:var(--muted); font-size:12px; margin-top:8px; }}
  .metrics {{ margin-top:28px; }}
  .metric-row {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }}
  .metric {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
    padding:16px; text-align:center; }}
  .m-val {{ font-size:28px; font-weight:700; color:var(--accent); }}
  .m-key {{ font-size:13px; margin-top:2px; }}
  .m-sub {{ font-size:11px; color:var(--muted); margin-top:3px; }}
  .stage {{ display:grid; grid-template-columns:1.4fr 1fr; gap:20px; align-items:center;
    background:var(--card); border:1px solid var(--line); border-radius:14px;
    padding:16px; margin:14px 0; }}
  .stage-img img {{ width:100%; border-radius:8px; display:block; }}
  .stage-img .missing {{ padding:40px; text-align:center; color:var(--muted);
    border:1px dashed var(--line); border-radius:8px; }}
  .stage-text h3 {{ margin:0 0 4px; font-size:17px; }}
  .stage-text .model {{ color:var(--accent); font-size:12px; margin-bottom:8px;
    font-family:ui-monospace,Menlo,Consolas,monospace; }}
  .stage-text p {{ margin:0; color:var(--muted); }}
  .panels {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
  .panel {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:16px; }}
  .panel h3 {{ margin:0 0 10px; font-size:15px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th, td {{ text-align:left; padding:5px 8px; border-bottom:1px solid var(--line); }}
  th {{ color:var(--muted); font-weight:600; }}
  .status {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:18px; }}
  .status ul {{ margin:6px 0 0; padding-left:18px; color:var(--muted); }}
  .status .works li {{ color:var(--fg); }}
  footer {{ margin-top:40px; color:var(--muted); font-size:12px;
    border-top:1px solid var(--line); padding-top:16px; }}
  @media (max-width:720px) {{ .stage,.panels,.status {{ grid-template-columns:1fr; }}
    .metric-row {{ grid-template-columns:1fr 1fr; }} }}
</style></head>
<body><div class="wrap">
  <header>
    <h1>{_esc(title)}</h1>
    <p>{_esc(subtitle)}</p>
    <div class="chips">
      <span class="chip">Python</span><span class="chip">PyTorch</span>
      <span class="chip">RF-DETR / Roboflow</span><span class="chip">ByteTrack</span>
      <span class="chip">OpenCV homography</span><span class="chip">supervision</span>
    </div>
  </header>
  {hero_html}

  <h2>The tracking pipeline</h2>
  {''.join(stage_html)}

  <h2>Extension in progress — shot detection &amp; box score</h2>
  <p class="intro">The tracking outputs above feed an experimental layer that
  detects shot attempts, calls make/miss and 2PT/3PT from court location, and
  aggregates a box score. This part is early-stage — shown as a direction, not a
  finished result.</p>
  <div class="panels">{shots_html}{box_html}</div>
  {metrics_html}

  <h2>Status & roadmap</h2>
  <div class="status">
    <div class="works"><strong>Working — player tracking</strong>
      <ul>
        <li>Per-frame detection of players, ball, rim, jersey numbers</li>
        <li>Multi-object tracking across frames (ByteTrack)</li>
        <li>Two-team appearance clustering</li>
        <li>Court homography → real-feet player positions & trajectories</li>
      </ul>
    </div>
    <div><strong>Next — shot-stats extension</strong>
      <ul>
        <li>Shot attempt + make/miss detection (raising precision)</li>
        <li>Jersey-number OCR for per-player box scores</li>
        <li>Free-throw detection (currently out of scope)</li>
        <li>Consistency across long uncut broadcasts</li>
      </ul>
    </div>
  </div>

  <footer>Generated by <code>hoopstats report</code> · extension metrics are on
  hand-annotated ground truth, field goals only, matched within ±2s.</footer>
</div></body></html>"""

    out_path.write_text(html, encoding="utf-8")
    return out_path
