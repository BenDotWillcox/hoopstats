"""Tests for segments loading, scope filtering, FT exclusion, and range I/O."""

import pytest

from hoopstats.evaluate import compute_metrics
from hoopstats.segments import (
    frame_in_segments,
    load_segments,
    merge_segments,
    segment_index,
    write_segments,
)


def write_csv(path, header, rows):
    lines = [header] + rows
    path.write_text("\n".join(lines) + "\n")
    return path


def pred(frame, result="make", team="0", number="23", shot_type="2PT"):
    return {
        "video_frame_idx": str(frame),
        "result": result,
        "offense_team_id": team,
        "shooter_number": number,
        "shot_type": shot_type,
    }


def label(frame, result="make", team="0", number="23", zone="paint"):
    return {
        "video_frame_idx": str(frame),
        "time_s": str(frame / 30.0),
        "result": result,
        "offense_team_id": team,
        "shooter_number": number,
        "zone": zone,
    }


class TestLoadSegments:
    def test_loads_all_rows_without_type(self, tmp_path):
        path = write_csv(tmp_path / "seg.csv", "start_frame,end_frame",
                         ["0,100", "200,300"])
        assert load_segments(path) == [(0, 100), (200, 300)]

    def test_filters_by_type(self, tmp_path):
        path = write_csv(tmp_path / "seg.csv", "start_frame,end_frame,type",
                         ["0,100,live", "100,200,replay", "200,300,live"])
        assert load_segments(path) == [(0, 100), (200, 300)]

    def test_custom_scope_types(self, tmp_path):
        path = write_csv(tmp_path / "seg.csv", "start_frame,end_frame,type",
                         ["0,100,live", "100,200,replay"])
        assert load_segments(path, scope_types=("replay",)) == [(100, 200)]

    def test_merges_overlapping(self, tmp_path):
        path = write_csv(tmp_path / "seg.csv", "start_frame,end_frame",
                         ["0,100", "50,150"])
        assert load_segments(path) == [(0, 150)]

    def test_rejects_inverted_range(self, tmp_path):
        path = write_csv(tmp_path / "seg.csv", "start_frame,end_frame", ["100,50"])
        with pytest.raises(ValueError):
            load_segments(path)


class TestWriteSegments:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "segs.csv"
        write_segments([(0, 100), (200, 300)], path)
        assert load_segments(path) == [(0, 100), (200, 300)]

    def test_write_merges_and_sorts(self, tmp_path):
        path = tmp_path / "segs.csv"
        write_segments([(200, 300), (0, 100), (50, 150)], path)
        assert load_segments(path) == [(0, 150), (200, 300)]

    def test_written_rows_are_tagged_live(self, tmp_path):
        path = tmp_path / "segs.csv"
        write_segments([(0, 100)], path)
        assert "0,100,live" in path.read_text()


class TestSegmentHelpers:
    def test_merge_adjacent(self):
        assert merge_segments([(0, 100), (100, 200)]) == [(0, 200)]

    def test_merge_disjoint_preserved(self):
        assert merge_segments([(200, 300), (0, 100)]) == [(0, 100), (200, 300)]

    def test_frame_in_segments_half_open(self):
        segs = [(0, 100)]
        assert frame_in_segments(0, segs)
        assert frame_in_segments(99, segs)
        assert not frame_in_segments(100, segs)  # end is exclusive

    def test_segment_index(self):
        segs = [(0, 100), (200, 300)]
        assert segment_index(50, segs) == 0
        assert segment_index(250, segs) == 1
        assert segment_index(150, segs) is None


class TestScopeFiltering:
    def test_out_of_scope_dropped_both_sides(self):
        preds = [pred(50), pred(500)]      # 500 is in a replay
        labels = [label(52), label(505)]   # 505 in a replay
        segments = [(0, 100)]
        m = compute_metrics(preds, labels, fps=30.0, segments=segments)
        assert m["scoped"] is True
        assert m["n_pred"] == 1
        assert m["n_label"] == 1
        assert m["n_matched"] == 1
        assert m["pred_dropped_out_of_scope"] == 1
        assert m["label_dropped_out_of_scope"] == 1
        assert m["precision"] == 1.0  # the replay false positive is excluded

    def test_no_segments_means_unscoped(self):
        m = compute_metrics([pred(50)], [label(52)], fps=30.0)
        assert m["scoped"] is False
        assert m["pred_dropped_out_of_scope"] == 0


class TestFreeThrowHandling:
    def test_ft_labels_excluded_from_fg_metrics(self):
        # One FG and one FT labeled; detector only catches the FG.
        preds = [pred(100)]
        labels = [label(100, zone="paint"), label(400, zone="ft")]
        m = compute_metrics(preds, labels, fps=30.0)
        assert m["n_label"] == 1          # FT not counted as a FG attempt
        assert m["ft_labeled"] == 1
        assert m["recall"] == 1.0         # the single FG was found
        assert m["ft_caught_by_detector"] == 0

    def test_pred_on_ft_not_a_false_positive(self):
        # Detector fires on a free throw; should not be penalized as FP.
        preds = [pred(400)]
        labels = [label(400, zone="ft")]
        m = compute_metrics(preds, labels, fps=30.0)
        assert m["ft_caught_by_detector"] == 1
        assert m["n_pred"] == 0            # the FT-matched pred leaves FG set
        assert m["false_positive_frames"] == []

    def test_ft_does_not_steal_nearby_fg_prediction(self):
        # A real FG prediction near a FG label still matches even with an FT around.
        preds = [pred(100)]
        labels = [label(100, zone="midrange"), label(900, zone="ft")]
        m = compute_metrics(preds, labels, fps=30.0)
        assert m["n_matched"] == 1
        assert m["ft_caught_by_detector"] == 0


class TestPermutationInvariantTeam:
    def test_flipped_team_labels_score_full(self):
        # Predicted teams are the consistent opposite of labels -> still 100%.
        preds = [pred(100, team="1"), pred(300, team="0")]
        labels = [label(100, team="0", zone="paint"),
                  label(300, team="1", zone="paint")]
        m = compute_metrics(preds, labels, fps=30.0)
        assert m["team_accuracy"] == 1.0

    def test_single_team_returns_none(self):
        # Only one team present -> metric is trivial, report None.
        preds = [pred(100, team="0")]
        labels = [label(100, team="0", zone="paint")]
        m = compute_metrics(preds, labels, fps=30.0)
        assert m["team_accuracy"] is None

    def test_genuine_team_error(self):
        # Predicted teams agree with neither alignment perfectly.
        preds = [pred(100, team="0"), pred(300, team="0"), pred(500, team="0")]
        labels = [label(100, team="0", zone="paint"),
                  label(300, team="1", zone="paint"),
                  label(500, team="1", zone="paint")]
        m = compute_metrics(preds, labels, fps=30.0)
        # best alignment: 2/3 (as-is gives 1/3, flipped gives 2/3)
        assert m["team_accuracy"] == pytest.approx(2 / 3)


class TestRangeIO:
    """Integration check against the bundled sample video, if present."""

    def test_iter_range_yields_absolute_indices(self):
        from pathlib import Path
        from hoopstats.video_io import VideoLoader

        video = Path("data/raw/bos-nyk.mp4")
        if not video.exists():
            pytest.skip("sample video not available")

        with VideoLoader(video) as loader:
            frames = list(loader.iter_range(50, 60))
        assert [idx for idx, _ in frames] == list(range(50, 60))
