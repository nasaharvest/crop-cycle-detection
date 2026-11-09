"""
Basic smoke test — runs crop_cycle_model on the bundled sample data
and verifies that at least one season is detected.

Usage:
    python test_basic.py
"""

import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from src.crop_cycle import (
    clean_ndvi_timeseries,
    crop_cycle_model,
    extract_seasons_trough_to_trough,
)


def test_sample_data():
    csv_path = os.path.join(os.path.dirname(__file__), "sample_data", "example_ndvi.csv")
    df = pd.read_csv(csv_path, parse_dates=["date"])
    assert not df.empty, "Sample CSV is empty"

    report, ndvi_processed = crop_cycle_model(df)

    assert not report.empty, (
        "crop_cycle_model returned an empty report — "
        "expected at least one season in the sample data."
    )
    assert "ndvi_peak_date" in report.columns
    assert "approx_season_start" in report.columns
    assert "approx_season_end" in report.columns
    assert not ndvi_processed.empty

    print(f"PASS — {len(report)} season(s) detected.")
    print(report[["season_number", "approx_season_start", "approx_season_end", "ndvi_peak_date"]].to_string(index=False))


def test_imports():
    from src.crop_cycle import crop_cycle_model  # noqa: F401
    from src.ndvi_preprocessing import whittaker_smooth, local_regression  # noqa: F401
    print("PASS — all imports resolved.")


def test_missing_ndvi_does_not_disable_outlier_removal():
    # Flat low NDVI, so the separate spike step (neighbours above 0.5) cannot be what removes the dip.
    dates = pd.date_range("2022-01-01", periods=100, freq="5D")
    ndvi = np.full(100, 0.3)
    ndvi[50] = 0.0
    ndvi[20] = np.nan
    cleaned = clean_ndvi_timeseries(pd.DataFrame({"date": dates, "NDVI": ndvi}))

    assert dates[50] not in set(cleaned["date"]), "cloud dip survived the outlier step"
    assert not cleaned["NDVI"].isna().any()
    print("PASS: a missing NDVI value leaves outlier removal working.")


def test_season_clipped_below_minimum_is_dropped():
    csv_path = os.path.join(os.path.dirname(__file__), "sample_data", "example_ndvi.csv")
    df = pd.read_csv(csv_path, parse_dates=["date"])

    # 66 days long before clipping, 51 days once it starts 3 days after the first season ends.
    starts = pd.to_datetime(["2022-03-01", "2022-05-20"])
    ends = pd.to_datetime(["2022-06-01", "2022-07-25"])
    peaks = pd.to_datetime(["2022-04-15", "2022-06-20"])
    rows = [
        pd.DataFrame([{
            "approx_season_start": starts[i],
            "approx_season_end": ends[i],
            "ndvi_peak_date": peaks[i],
            "ndvi_peak_value": 0.8,
            "trough_before_peak": pd.NaT,
            "trough_after_peak": pd.NaT,
            "troughs_days_distance": np.nan,
        }])
        for i in range(2)
    ]
    cycles = pd.DataFrame({"ndvi_peak_value": [0.8, 0.8]})
    with patch("src.crop_cycle.extract_seasons_trough_to_trough", return_value=cycles), \
            patch("src.crop_cycle.process_single_crop_cycle", side_effect=rows):
        report, _ = crop_cycle_model(df, min_season_length=60)

    assert len(report) == 1, report
    assert report["season_number"].tolist() == [1]
    print("PASS: a season clipped below the minimum is dropped.")


def test_last_orphan_peak_is_rounded():
    extrema = pd.DataFrame({
        "NDVI": [0.2, 0.7777],
        "date": pd.to_datetime(["2022-05-01", "2022-08-01"]),
        "DOY": [121, 213],
        "type": ["trough", "peak"],
    })
    seasons = extract_seasons_trough_to_trough(extrema)
    assert seasons["ndvi_peak_value"].tolist() == [0.78]
    print("PASS: peak after the last trough is rounded like the others.")


if __name__ == "__main__":
    print("Running import test ...")
    test_imports()
    print("\nRunning sample data test ...")
    test_sample_data()
    print("\nRunning regression tests ...")
    test_missing_ndvi_does_not_disable_outlier_removal()
    test_season_clipped_below_minimum_is_dropped()
    test_last_orphan_peak_is_rounded()
    print("\nAll tests passed.")
