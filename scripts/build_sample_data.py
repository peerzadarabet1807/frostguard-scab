"""Regenerate the sample datasets shipped in ``data/``.

* ``data/shopian_spring_2024_hourly.csv`` — real ERA5 reanalysis for Shopian
  (1 Apr - 15 May 2024, the primary apple scab season) from the Open-Meteo archive.
* ``data/mock_<scenario>.csv`` — deterministic synthetic scenarios for offline demos.
* ``data/sample_leaf_scab.jpg`` / ``sample_leaf_healthy.jpg`` — procedural test leaves
  (+ ``sample_leaf_scab_boxes.json`` ground truth).

Usage::

    python scripts/build_sample_data.py            # everything
    python scripts/build_sample_data.py --skip-archive
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.weather_fetcher import (  # noqa: E402
    HOURLY_VARIABLES,
    MOCK_SCENARIOS,
    SHOPIAN,
    WeatherFetcher,
    generate_mock_weather,
)
from src.vision.synthetic import render_healthy_leaf, render_synthetic_leaf  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data"
logger = logging.getLogger("build_sample_data")


def build_archive_csv(attempts: int = 5) -> Path | None:
    out = DATA_DIR / "shopian_spring_2024_hourly.csv"
    fetcher = WeatherFetcher(cache_dir=PROJECT_ROOT / ".cache", timeout_seconds=20.0)
    for attempt in range(1, attempts + 1):
        try:
            df = fetcher.fetch_historical(SHOPIAN.latitude, SHOPIAN.longitude, "2024-04-01", "2024-05-15")
        except Exception as exc:  # noqa: BLE001
            logger.warning("archive attempt %d/%d failed: %s", attempt, attempts, exc)
            time.sleep(2 * attempt)
            continue
        df.round(dict.fromkeys(HOURLY_VARIABLES, 3)).to_csv(out, index=False)
        logger.info("wrote %s (%d rows)", out.relative_to(PROJECT_ROOT), len(df))
        return out
    logger.error("Open-Meteo archive unreachable; kept existing %s", out.name)
    return None


def build_mock_csvs() -> list[Path]:
    paths = []
    for scenario in sorted(MOCK_SCENARIOS):
        out = DATA_DIR / f"mock_{scenario}.csv"
        generate_mock_weather(start="2024-04-12 00:00", hours=96, scenario=scenario).to_csv(out, index=False)
        logger.info("wrote %s", out.relative_to(PROJECT_ROOT))
        paths.append(out)
    return paths


def build_leaf_images() -> None:
    scab, boxes = render_synthetic_leaf(seed=7)
    scab.save(DATA_DIR / "sample_leaf_scab.jpg", quality=92)
    (DATA_DIR / "sample_leaf_scab_boxes.json").write_text(
        json.dumps({"label": "apple_scab", "boxes_xyxy": boxes}, indent=2) + "\n", encoding="utf-8"
    )
    render_healthy_leaf(seed=7).save(DATA_DIR / "sample_leaf_healthy.jpg", quality=92)
    logger.info("wrote synthetic leaf images (%d lesions)", len(boxes))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-archive", action="store_true", help="do not call the Open-Meteo archive API")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    DATA_DIR.mkdir(exist_ok=True)
    build_mock_csvs()
    build_leaf_images()
    if not args.skip_archive:
        build_archive_csv()


if __name__ == "__main__":
    main()
