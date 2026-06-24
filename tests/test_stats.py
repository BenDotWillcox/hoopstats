"""Unit tests for shots.csv / box_score.csv export."""

import csv

import pytest

from hoopstats.models import ShotEvent
from hoopstats.stats import export_box_score_csv, export_shots_csv


def shot(team="0", number="23", result="make", shot_type="2PT", frame=10, **kwargs):
    defaults = dict(
        period=1,
        game_clock_s=600.0,
        offense_team_id=team,
        defense_team_id=None,
        shooter_global_id=f"{team}#{number}" if team and number else None,
        shooter_number=number,
        result=result,
        shot_type=shot_type,
        x_ft=10.0,
        y_ft=25.0,
        distance_ft=4.8,
        video_frame_idx=frame,
        shooter_xyxy=(100, 100, 150, 200),
    )
    defaults.update(kwargs)
    return ShotEvent(**defaults)


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


class TestShotsCsv:
    def test_writes_one_row_per_shot_without_internal_fields(self, tmp_path):
        path = export_shots_csv([shot(), shot(result="miss", frame=50)], tmp_path)
        rows = read_csv(path)
        assert len(rows) == 2
        assert "shooter_xyxy" not in rows[0]
        assert rows[0]["result"] == "make"
        assert rows[1]["result"] == "miss"
        assert rows[0]["shooter_number"] == "23"
        assert rows[0]["x_ft"] == "10.0"

    def test_empty_shot_list_still_writes_header(self, tmp_path):
        path = export_shots_csv([], tmp_path)
        rows = read_csv(path)
        assert rows == []
        assert path.read_text().startswith("period,")


class TestBoxScoreCsv:
    def test_aggregates_per_player(self, tmp_path):
        shots = [
            shot(number="23", result="make", shot_type="2PT", frame=10),
            shot(number="23", result="make", shot_type="3PT", frame=50),
            shot(number="23", result="miss", shot_type="2PT", frame=90),
            shot(number="7", team="1", result="miss", shot_type="3PT", frame=130),
        ]
        shots_csv = export_shots_csv(shots, tmp_path)
        box_csv = export_box_score_csv(shots_csv, tmp_path)
        rows = {r["player"]: r for r in read_csv(box_csv)}

        p23 = rows["0#23"]
        assert p23["fga"] == "3"
        assert p23["fgm"] == "2"
        assert p23["tpa"] == "1"
        assert p23["tpm"] == "1"
        assert p23["pts"] == "5"  # one 2PT + one 3PT
        assert float(p23["fg_pct"]) == pytest.approx(0.667)

        p7 = rows["1#7"]
        assert p7["fga"] == "1"
        assert p7["tpa"] == "1"
        assert p7["pts"] == "0"

    def test_unknown_shooter_bucketed_by_team(self, tmp_path):
        shots = [
            shot(number=None, shooter_global_id=None, result="make"),
            shot(number=None, team=None, shooter_global_id=None, result="miss", frame=60),
        ]
        shots_csv = export_shots_csv(shots, tmp_path)
        box_csv = export_box_score_csv(shots_csv, tmp_path)
        rows = {r["player"]: r for r in read_csv(box_csv)}

        assert rows["0#unknown"]["pts"] == "2"
        assert rows["?#unknown"]["fga"] == "1"

    def test_team_totals_reconcile_with_shot_log(self, tmp_path):
        shots = [shot(number=str(n % 3), result="make" if n % 2 else "miss",
                      frame=n * 40) for n in range(6)]
        shots_csv = export_shots_csv(shots, tmp_path)
        box_csv = export_box_score_csv(shots_csv, tmp_path)
        rows = read_csv(box_csv)
        assert sum(int(r["fga"]) for r in rows) == 6
        assert sum(int(r["fgm"]) for r in rows) == 3
