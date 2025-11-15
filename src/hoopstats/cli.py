from pathlib import Path
import argparse

from .pipeline import GameProcessor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process a basketball game into shots + box score.")
    parser.add_argument("--video", required=True, help="Path to game video")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    gp = GameProcessor(Path(args.video), Path(args.out))
    gp.run()
