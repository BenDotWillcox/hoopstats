"""Tests for the portfolio HTML report generator."""

import numpy as np
import pytest

from hoopstats.report import generate_pipeline_html


def _write_jpg(path, w=64, h=36):
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.zeros((h, w, 3), dtype=np.uint8))


METRICS = {
    "f1": 0.545, "precision": 0.448, "recall": 0.698,
    "result_accuracy": 0.767, "result_correct": 23, "result_n": 30,
    "shot_type_accuracy": 0.733, "shot_type_correct": 22, "shot_type_n": 30,
    "team_accuracy": 0.633, "n_label": 43,
}


def test_missing_images_render_placeholders(tmp_path):
    out = generate_pipeline_html(
        tmp_path / "p.html", tmp_path / "empty", shots=[], box_score=[], metrics=None)
    html = out.read_text(encoding="utf-8")
    assert "artifact not generated" in html
    assert "data:image/jpeg;base64" not in html


def test_embeds_found_images(tmp_path):
    frames = tmp_path / "frames"
    _write_jpg(frames / "single_frame_detection" / "detect_frame_0.jpg")
    _write_jpg(frames / "keypoint_detection" / "keypoints_frame_0.jpg")
    out = generate_pipeline_html(
        tmp_path / "p.html", frames, shots=[], box_score=[], metrics=None)
    html = out.read_text(encoding="utf-8")
    assert html.count("data:image/jpeg;base64") == 2  # only the two provided


def test_metrics_hidden_by_default(tmp_path):
    out = generate_pipeline_html(
        tmp_path / "p.html", tmp_path / "f", shots=[], box_score=[], metrics=METRICS)
    html = out.read_text(encoding="utf-8")
    assert "Accuracy so far" not in html
    assert "55%" not in html


def test_metrics_strip_rendered_when_opted_in(tmp_path):
    out = generate_pipeline_html(
        tmp_path / "p.html", tmp_path / "f", shots=[], box_score=[],
        metrics=METRICS, show_metrics=True)
    html = out.read_text(encoding="utf-8")
    assert "55%" in html  # f1
    assert "77%" in html  # make/miss
    assert "vs 43 hand-labeled" in html


def test_pipeline_precedes_metrics(tmp_path):
    """Tracking pipeline is shown first; accuracy metrics moved to the bottom."""
    out = generate_pipeline_html(
        tmp_path / "p.html", tmp_path / "f", shots=[], box_score=[],
        metrics=METRICS, show_metrics=True)
    html = out.read_text(encoding="utf-8")
    assert html.index("The tracking pipeline") < html.index("Accuracy so far")
    assert html.index("Extension in progress") < html.index("Accuracy so far")


def test_box_score_excludes_unknown_and_ranks(tmp_path):
    box = [
        {"player": "0#11", "team": "0", "fgm": "3", "fga": "7", "tpm": "0", "tpa": "1", "pts": "6"},
        {"player": "1#unknown", "team": "1", "fgm": "3", "fga": "11", "tpm": "3", "tpa": "10", "pts": "9"},
        {"player": "1#8", "team": "1", "fgm": "1", "fga": "5", "tpm": "1", "tpa": "2", "pts": "3"},
    ]
    out = generate_pipeline_html(
        tmp_path / "p.html", tmp_path / "f", shots=[], box_score=box, metrics=None)
    html = out.read_text(encoding="utf-8")
    assert "0#11" in html
    assert "#unknown" not in html  # unknown bucket excluded from top scorers


def test_videos_embedded_via_relative_path(tmp_path):
    src = tmp_path / "clip-court-map.mp4"
    src.write_bytes(b"\x00\x00fake mp4")
    out = generate_pipeline_html(
        tmp_path / "report" / "p.html", tmp_path / "f", shots=[], box_score=[],
        metrics=None, videos=[{"path": str(src), "caption": "top-down"}])
    html = out.read_text(encoding="utf-8")
    assert "See it track" in html
    assert 'src="assets/clip-court-map.mp4"' in html
    # video copied next to the HTML
    assert (out.parent / "assets" / "clip-court-map.mp4").exists()


def test_missing_video_skipped(tmp_path):
    out = generate_pipeline_html(
        tmp_path / "p.html", tmp_path / "f", shots=[], box_score=[],
        metrics=None, videos=[{"path": str(tmp_path / "nope.mp4"), "caption": "x"}])
    html = out.read_text(encoding="utf-8")
    assert "See it track" not in html


def test_shot_log_preview(tmp_path):
    shots = [{"video_frame_idx": "754", "result": "make", "shot_type": "2PT",
              "offense_team_id": "1", "shooter_number": "11", "distance_ft": "19.5"}]
    out = generate_pipeline_html(
        tmp_path / "p.html", tmp_path / "f", shots=shots, box_score=[], metrics=None)
    html = out.read_text(encoding="utf-8")
    assert "1 detected shots" in html
    assert "754" in html
