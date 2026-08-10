"""
Crop cycle detection from an NDVI time series.

Main entry point
----------------
    crop_cycle_model(df) -> (report, ndvi_for_processing)

``df`` must contain at least two columns: ``date`` (datetime-like) and
``NDVI`` (float).  An optional ``poly_name`` column identifies the field;
if absent a default name is used.

The function returns a per-season milestone table and the daily
interpolated + smoothed NDVI series used internally.
"""

# Standard library
from typing import Tuple
import logging

# Third-party
import numpy as np
import pandas as pd
import pwlf
from scipy.signal import argrelmax, argrelmin
from statsmodels.tsa.filters.hp_filter import hpfilter

# Local
from .ndvi_preprocessing import local_regression, whittaker_smooth


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NDVI cleaning
# ---------------------------------------------------------------------------

def _detect_downward_spikes(
    df: pd.DataFrame,
    max_days_for_spikes: int,
    NDVI_spike_threshold: float,
    min_NDVI_for_spike: float,
) -> pd.DataFrame:
    """
    Detect downward NDVI spikes using a three-point (t-1, t, t+1) window.

    A spike is a sharp drop followed by an equally sharp recovery, where
    both neighbours exceed ``min_NDVI_for_spike`` and the window width is
    within ``max_days_for_spikes``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``date`` and ``NDVI``.
    max_days_for_spikes : int
        Maximum time span (days) between t-1 and t+1.
    NDVI_spike_threshold : float
        Minimum absolute NDVI change to qualify as a spike.
    min_NDVI_for_spike : float
        Minimum NDVI at t-1 and t+1 for detection to apply.

    Returns
    -------
    pd.DataFrame
        Rows of ``df`` flagged as downward spikes.
    """
    spikes = df[["date", "NDVI"]].copy()
    spikes.rename(columns={"NDVI": "NDVI_t1"}, inplace=True)
    spikes["NDVI_t0"] = spikes["NDVI_t1"].shift(1)
    spikes["NDVI_t2"] = spikes["NDVI_t1"].shift(-1)
    spikes["NDVI_date_t0"] = spikes["date"].shift(1)
    spikes["NDVI_date_t2"] = spikes["date"].shift(-1)
    spikes["date_diff"] = (spikes["NDVI_date_t2"] - spikes["NDVI_date_t0"]).dt.days
    spikes = spikes[spikes["date_diff"] <= max_days_for_spikes].copy()
    spikes.dropna(inplace=True)

    c1 = (spikes["NDVI_t0"] - spikes["NDVI_t1"]) > NDVI_spike_threshold
    c2 = (spikes["NDVI_t1"] - spikes["NDVI_t2"]) < -NDVI_spike_threshold
    c3 = (spikes["NDVI_t0"] > min_NDVI_for_spike) & (spikes["NDVI_t2"] > min_NDVI_for_spike)
    return spikes[c1 & c2 & c3].copy()


def clean_ndvi_timeseries(
    df: pd.DataFrame,
    k_to_find_outliers: int = 4,
    dev_threshold_lower: float = -0.015,
    max_days_for_spikes: int = 30,
    NDVI_spike_threshold: float = 0.06,
    min_NDVI_for_spike: float = 0.5,
) -> pd.DataFrame:
    """
    Remove cloud-related outliers and downward spikes from an NDVI series.

    Steps:

    1. Ensure a ``poly_name`` column exists.
    2. Sort by ``[poly_name, date]``.
    3. Smooth with local regression (window ``k_to_find_outliers``).
    4. Drop observations whose signed deviation from the smooth falls below
       ``dev_threshold_lower``.
    5. Detect and remove three-point downward spikes.

    Parameters
    ----------
    df : pd.DataFrame
        Columns: ``date`` (datetime64), ``NDVI`` (float).
        Optional: ``poly_name`` (str).  Data must be for a single polygon.
    k_to_find_outliers : int, optional
        Local-regression window size. Default 4.
    dev_threshold_lower : float, optional
        Signed-deviation cut-off for outlier removal. Default -0.015.
    max_days_for_spikes : int, optional
        Maximum three-point window width (days) for spike detection. Default 30.
    NDVI_spike_threshold : float, optional
        Minimum NDVI drop to qualify as a spike. Default 0.06.
    min_NDVI_for_spike : float, optional
        Minimum NDVI at neighbours for spike detection. Default 0.5.

    Returns
    -------
    pd.DataFrame
        Columns: ``poly_name``, ``date``, ``NDVI``.
    """
    if df.empty:
        return df

    df = df.copy()
    if "poly_name" not in df.columns:
        df["poly_name"] = "unnamed polygon"

    df.sort_values(["poly_name", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    x = (df["date"] - df["date"].min()).dt.days.tolist()
    y = df["NDVI"].tolist()
    df["NDVI_smooth"] = [local_regression(x[p], x, y, k_to_find_outliers) for p in range(len(x))]
    df["squared_error"] = (df["NDVI"] - df["NDVI_smooth"]) ** 2
    df["signed_error"] = df["squared_error"] * np.where(
        (df["NDVI"] - df["NDVI_smooth"]) > 0, 1, -1
    )

    outliers = df[df["signed_error"] <= dev_threshold_lower].copy()
    df_no_outliers = df[~df["date"].isin(outliers["date"])].copy()

    downward_spikes = _detect_downward_spikes(
        df_no_outliers,
        max_days_for_spikes=max_days_for_spikes,
        NDVI_spike_threshold=NDVI_spike_threshold,
        min_NDVI_for_spike=min_NDVI_for_spike,
    )
    df_no_spikes = df_no_outliers[~df_no_outliers["date"].isin(downward_spikes["date"])].copy()
    return df_no_spikes[["poly_name", "date", "NDVI"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Peak / trough validation
# ---------------------------------------------------------------------------

def check_peak_validity(
    max_value: float,
    max_date: pd.Timestamp,
    min_ndvi_for_peak: float = 0.5,
) -> pd.DataFrame:
    """
    Validate a candidate NDVI peak against a minimum threshold.

    Parameters
    ----------
    max_value : float
        NDVI value at the candidate peak.
    max_date : pd.Timestamp
        Date of the candidate peak.
    min_ndvi_for_peak : float, optional
        Minimum NDVI required for a valid peak. Default 0.5.

    Returns
    -------
    pd.DataFrame
        One-row DataFrame ``[NDVI, date, DOY]`` if valid, else empty.
    """
    if max_date is None or pd.isna(max_date):
        return pd.DataFrame()
    if not isinstance(max_date, pd.Timestamp):
        try:
            max_date = pd.to_datetime(max_date)
        except Exception as e:
            logger.error(f"check_peak_validity: cannot parse date ({max_date}): {e}")
            return pd.DataFrame()

    if max_value > min_ndvi_for_peak:
        return pd.DataFrame(
            [[max_value, max_date.date(), max_date.dayofyear]],
            columns=["NDVI", "date", "DOY"],
        )
    return pd.DataFrame()


def check_trough_validity(
    trough_value: float,
    trough_date: pd.Timestamp,
    max_ndvi_for_trough: float = 0.5,
) -> pd.DataFrame:
    """
    Validate a candidate NDVI trough against a maximum threshold.

    Parameters
    ----------
    trough_value : float
        NDVI value at the candidate trough.
    trough_date : pd.Timestamp
        Date of the candidate trough.
    max_ndvi_for_trough : float, optional
        Maximum NDVI for a valid trough. Default 0.5.

    Returns
    -------
    pd.DataFrame
        One-row DataFrame ``[NDVI, date, DOY]`` if valid, else empty.
    """
    if pd.isnull(trough_value) or pd.isnull(trough_date):
        return pd.DataFrame()
    if not isinstance(trough_date, pd.Timestamp):
        try:
            trough_date = pd.to_datetime(trough_date)
        except Exception:
            return pd.DataFrame()

    if trough_value < max_ndvi_for_trough:
        return pd.DataFrame(
            [[trough_value, trough_date.date(), trough_date.dayofyear]],
            columns=["NDVI", "date", "DOY"],
        )
    return pd.DataFrame()


def validate_peaks_and_troughs(
    ndvi_series: pd.Series,
    date_series: pd.Series,
    min_ndvi_for_peak: float = 0.5,
) -> pd.DataFrame:
    """
    Identify and validate all local peaks and troughs in an NDVI series.

    Parameters
    ----------
    ndvi_series : pd.Series
        Smoothed, interpolated NDVI values.
    date_series : pd.Series
        Corresponding dates.
    min_ndvi_for_peak : float, optional
        Minimum NDVI for a valid peak. Default 0.5.

    Returns
    -------
    pd.DataFrame
        Columns: ``NDVI``, ``date``, ``DOY``, ``type`` (``'peak'`` or ``'trough'``).
    """
    peaks_idx = argrelmax(ndvi_series.values, order=5)[0]
    troughs_idx = argrelmin(ndvi_series.values, order=5)[0]

    peaks_rows = [
        check_peak_validity(ndvi_series[p], date_series[p], min_ndvi_for_peak)
        for p in peaks_idx
    ]
    peaks_df = pd.concat(peaks_rows, axis=0).dropna() if peaks_rows else pd.DataFrame()
    if not peaks_df.empty:
        peaks_df["type"] = "peak"

    troughs_rows = [
        check_trough_validity(ndvi_series[t], date_series[t])
        for t in troughs_idx
    ]
    troughs_df = pd.concat(troughs_rows, axis=0).dropna() if troughs_rows else pd.DataFrame()
    if not troughs_df.empty:
        troughs_df["type"] = "trough"

    if peaks_df.empty and troughs_df.empty:
        return pd.DataFrame()
    return pd.concat([peaks_df, troughs_df]).reset_index(drop=True)


def detect_peaks_n_troughs(
    ndvi_series: pd.Series,
    dates_series: pd.Series,
    min_ndvi_for_peak: float = 0.5,
) -> pd.DataFrame:
    """
    Thin wrapper around ``validate_peaks_and_troughs`` with error handling.

    Parameters
    ----------
    ndvi_series : pd.Series
        Smoothed, interpolated NDVI.
    dates_series : pd.Series
        Corresponding dates.
    min_ndvi_for_peak : float, optional
        Minimum NDVI for a valid peak. Default 0.5.

    Returns
    -------
    pd.DataFrame
        Peaks and troughs with columns ``NDVI``, ``date``, ``DOY``, ``type``.
        Empty DataFrame on error or if no valid extrema are found.
    """
    try:
        pt = validate_peaks_and_troughs(ndvi_series, dates_series, min_ndvi_for_peak)
        if not pt.empty:
            pt["date"] = pd.to_datetime(pt["date"])
        return pt
    except Exception as e:
        logger.error(f"Error in detect_peaks_n_troughs: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Season extraction
# ---------------------------------------------------------------------------

def extract_seasons_trough_to_trough(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group peaks and troughs into trough-to-trough seasons.

    For each consecutive trough pair the highest-NDVI peak between them is
    selected.  "Orphan" peaks before the first trough or after the last
    trough are also captured.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``detect_peaks_n_troughs``:
        columns ``NDVI``, ``date``, ``DOY``, ``type``.

    Returns
    -------
    pd.DataFrame
        Columns: ``season_number``, ``trough_before_peak``,
        ``trough_after_peak``, ``troughs_days_distance``,
        ``ndvi_peak_value``, ``ndvi_peak_date``.
    """
    df = df.sort_values("date").reset_index(drop=True)
    peaks = df[df["type"] == "peak"].copy()
    troughs = df[df["type"] == "trough"].copy()
    seasons = []

    for i in range(len(troughs) - 1):
        t_start = troughs.iloc[i]
        t_end = troughs.iloc[i + 1]
        peaks_between = peaks[
            (peaks["date"] > t_start["date"]) & (peaks["date"] < t_end["date"])
        ]
        if not peaks_between.empty:
            best = peaks_between.loc[peaks_between["NDVI"].idxmax()]
            seasons.append({
                "trough_before_peak": t_start["date"],
                "trough_after_peak": t_end["date"],
                "troughs_days_distance": (t_end["date"] - t_start["date"]).days,
                "ndvi_peak_value": round(best["NDVI"], 2),
                "ndvi_peak_date": best["date"],
            })

    first_trough = troughs.iloc[0]["date"] if not troughs.empty else pd.NaT
    for _, peak in peaks[peaks["date"] < first_trough].iterrows():
        seasons.append({
            "trough_before_peak": pd.NaT,
            "trough_after_peak": first_trough,
            "troughs_days_distance": np.nan,
            "ndvi_peak_value": round(peak["NDVI"], 2),
            "ndvi_peak_date": peak["date"],
        })

    last_trough = troughs.iloc[-1]["date"] if not troughs.empty else pd.NaT
    for _, peak in peaks[peaks["date"] > last_trough].iterrows():
        seasons.append({
            "trough_before_peak": last_trough,
            "trough_after_peak": pd.NaT,
            "troughs_days_distance": np.nan,
            "ndvi_peak_value": peak["NDVI"],
            "ndvi_peak_date": peak["date"],
        })

    if not seasons:
        return pd.DataFrame(
            columns=[
                "season_number", "trough_before_peak", "trough_after_peak",
                "troughs_days_distance", "ndvi_peak_value", "ndvi_peak_date",
            ]
        )
    seasons_df = (
        pd.DataFrame(seasons)
        .sort_values("ndvi_peak_date")
        .reset_index(drop=True)
    )
    seasons_df.insert(0, "season_number", range(1, len(seasons_df) + 1))
    return seasons_df


# ---------------------------------------------------------------------------
# Season processing
# ---------------------------------------------------------------------------

def estimate_season_start_end(
    row: pd.Series,
    df: pd.DataFrame,
    ndvi_threshold_for_approx_season_start: float,
    ndvi_threshold_for_approx_season_end: float,
    adaptive_start_threshold: float = None,
    adaptive_end_threshold: float = None,
    season_window_days: int = 90,
) -> pd.Series:
    """
    Estimate approximate start and end dates of a season from NDVI thresholds.

    The start is the last date before the peak where
    ``NDVI_interpolated <= threshold``; the end is the first date after the
    peak where ``NDVI_interpolated <= threshold``.

    Adaptive thresholds (derived from peak NDVI) are tried first.  If the
    curve never crosses them within the search window, fixed fallback
    thresholds are used.  If neither is crossed, trough-based fallbacks apply.

    Parameters
    ----------
    row : pd.Series
        Season row with keys ``trough_before_peak``, ``ndvi_peak_date``,
        ``trough_after_peak``.
    df : pd.DataFrame
        Daily interpolated NDVI (columns ``date``, ``NDVI_interpolated``).
    ndvi_threshold_for_approx_season_start : float
        Fixed NDVI threshold for season start.
    ndvi_threshold_for_approx_season_end : float
        Fixed NDVI threshold for season end.
    adaptive_start_threshold : float, optional
        Peak-relative NDVI threshold tried first for the start.
    adaptive_end_threshold : float, optional
        Peak-relative NDVI threshold tried first for the end.
    season_window_days : int, optional
        Search window (days) when no bounding trough exists. Default 90.

    Returns
    -------
    pd.Series
        Keys: ``approx_season_start``, ``approx_season_end``.
    """
    peak_date = row["ndvi_peak_date"]
    trough_before = row.get("trough_before_peak")
    trough_after = row.get("trough_after_peak")

    start_window_min = (
        trough_before if pd.notna(trough_before)
        else peak_date - pd.Timedelta(days=season_window_days)
    )
    before_peak = df[(df["date"] >= start_window_min) & (df["date"] < peak_date)]

    approx_season_start = pd.NaT
    for thresh in [t for t in [adaptive_start_threshold, ndvi_threshold_for_approx_season_start] if t is not None]:
        approx_season_start = before_peak[before_peak["NDVI_interpolated"] <= thresh]["date"].max()
        if pd.notna(approx_season_start):
            break
    if pd.isna(approx_season_start) and pd.notna(trough_before):
        approx_season_start = trough_before + pd.Timedelta(days=3)

    end_window_max = (
        trough_after if pd.notna(trough_after)
        else peak_date + pd.Timedelta(days=season_window_days)
    )
    after_peak = df[(df["date"] > peak_date) & (df["date"] <= end_window_max)]

    approx_season_end = pd.NaT
    for thresh in [t for t in [adaptive_end_threshold, ndvi_threshold_for_approx_season_end] if t is not None]:
        approx_season_end = after_peak[after_peak["NDVI_interpolated"] <= thresh]["date"].min()
        if pd.notna(approx_season_end):
            break
    if pd.isna(approx_season_end) and pd.notna(trough_after):
        approx_season_end = trough_after - pd.Timedelta(days=3)

    return pd.Series({
        "approx_season_start": approx_season_start,
        "approx_season_end": approx_season_end,
    })


def get_dates_for_season_pwlf(
    row: pd.Series,
    df: pd.DataFrame,
    season_type: str,
    ndvi_threshold_to_extent_data_for_pwlf: float,
    season_n_days_buffer: int,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Determine the extended date window used for piecewise linear fitting.

    Parameters
    ----------
    row : pd.Series
        Season row with keys ``trough_before_peak``, ``ndvi_peak_date``,
        ``trough_after_peak``.
    df : pd.DataFrame
        Daily interpolated NDVI (columns ``date``, ``NDVI_interpolated``).
    season_type : str
        One of ``'two_troughs'``, ``'only_1st_trough'``, ``'only_2nd_trough'``.
    ndvi_threshold_to_extent_data_for_pwlf : float
        NDVI threshold for extending the window boundaries.
    season_n_days_buffer : int
        Days added around the trough when extending.

    Returns
    -------
    Tuple[pd.Timestamp, pd.Timestamp]
        ``(season_ext_before, season_ext_after)``
    """
    valid_types = ("two_troughs", "only_1st_trough", "only_2nd_trough")
    if season_type not in valid_types:
        raise ValueError(f"Invalid season_type '{season_type}'. Must be one of {valid_types}.")

    dates = df["date"]
    ndvi = df["NDVI_interpolated"]

    if season_type in ("two_troughs", "only_1st_trough"):
        mask = (
            dates.between(
                row["trough_before_peak"] - pd.DateOffset(days=season_n_days_buffer),
                row["trough_before_peak"],
            )
            & (ndvi < ndvi_threshold_to_extent_data_for_pwlf)
        )
        season_ext_before = dates[mask].min()
        if pd.isna(season_ext_before):
            season_ext_before = row["trough_before_peak"]
    else:
        season_ext_before = row["ndvi_peak_date"] - pd.DateOffset(days=season_n_days_buffer)

    if season_type in ("two_troughs", "only_2nd_trough"):
        mask = (
            dates.between(
                row["trough_after_peak"],
                row["trough_after_peak"] + pd.DateOffset(days=season_n_days_buffer),
            )
            & (ndvi < ndvi_threshold_to_extent_data_for_pwlf)
        )
        season_ext_after = dates[mask].max()
        if pd.isna(season_ext_after):
            season_ext_after = row["trough_after_peak"]
    else:
        season_ext_after = row["ndvi_peak_date"] + pd.DateOffset(days=season_n_days_buffer)

    return season_ext_before, season_ext_after


def get_dates_for_greenup(
    row: pd.Series,
    df: pd.DataFrame,
    greenup_min_ndvi_threshold: float = 0.3,
    greenup_max_ndvi_threshold: float = 0.6,
) -> tuple:
    """
    Estimate the start and end of the green-up phase from NDVI thresholds.

    Parameters
    ----------
    row : pd.Series
        Must contain ``approx_season_start`` and ``ndvi_peak_date``.
    df : pd.DataFrame
        Daily interpolated NDVI (columns ``date``, ``NDVI_interpolated``).
    greenup_min_ndvi_threshold : float, optional
        NDVI at the beginning of green-up. Default 0.3.
    greenup_max_ndvi_threshold : float, optional
        NDVI at the end of green-up. Default 0.6.

    Returns
    -------
    tuple
        ``(greenup_start_date, greenup_end_date)``
    """
    window = df[df["date"].between(row.get("approx_season_start"), row.get("ndvi_peak_date"))]

    greenup_start = window[window["NDVI_interpolated"] <= greenup_min_ndvi_threshold]["date"].max()
    if pd.isna(greenup_start):
        greenup_start = row.get("approx_season_start")

    greenup_end = window[window["NDVI_interpolated"] <= greenup_max_ndvi_threshold]["date"].max()
    return greenup_start, greenup_end


def detect_greenup(df: pd.DataFrame, greenup_ndvi_max: float = 0.5):
    """
    Detect the green-up date as the point of maximum positive NDVI gradient.

    Parameters
    ----------
    df : pd.DataFrame
        Columns ``date`` and ``NDVI_interpolated`` (daily).
    greenup_ndvi_max : float, optional
        Upper NDVI bound for green-up detection. Default 0.5.

    Returns
    -------
    pandas.Timestamp or None
        Detected green-up date, or ``None`` if not found.
    """
    if len(df) < 2:
        return None
    ndvi = df["NDVI_interpolated"].to_numpy()
    dates = df["date"].to_numpy()
    dNDVI = np.gradient(ndvi)
    mask = (ndvi >= 0) & (ndvi <= greenup_ndvi_max) & (dNDVI > 0)
    if np.any(mask):
        idx = np.argmax(dNDVI[mask])
        return dates[np.where(mask)[0][idx]]
    return None


def piecewise_ndvi_breakpoints(df: pd.DataFrame, n_segments: int = 5) -> dict:
    """
    Fit a piecewise linear model to a seasonal NDVI window and return breakpoints.

    Parameters
    ----------
    df : pd.DataFrame
        Columns ``date`` and ``NDVI_interpolated`` for the season window.
    n_segments : int, optional
        Number of linear segments. Default 5.

    Returns
    -------
    dict
        Keys: ``break_dates`` (list of Timestamps), ``break_doy`` (list of int),
        ``df_fit`` (DataFrame with ``date``, ``NDVI_interpolated``, ``NDVI_fit``).
    """
    x = (df["date"] - df["date"].min()).dt.days.values
    y = df["NDVI_interpolated"].values
    model = pwlf.PiecewiseLinFit(x, y)
    breakpoints = model.fit(n_segments)
    x_pred = np.linspace(x.min(), x.max(), len(x))
    y_pred = model.predict(x_pred)
    df_fit = pd.DataFrame({
        "date": df["date"].min() + pd.to_timedelta(x_pred, unit="D"),
        "NDVI_fit": y_pred,
    })
    df_out = df.merge(df_fit, on="date", how="left")
    break_dates = [df["date"].min() + pd.Timedelta(days=float(b)) for b in breakpoints]
    break_doy = [d.timetuple().tm_yday for d in break_dates]
    return {"break_dates": break_dates, "break_doy": break_doy, "df_fit": df_out}


def detect_breakpoints_in_season(
    season_ndvi_for_pwlf: pd.DataFrame,
    season_type: str,
) -> pd.DataFrame:
    """
    Run piecewise linear fitting and return named breakpoint dates.

    Parameters
    ----------
    season_ndvi_for_pwlf : pd.DataFrame
        Season NDVI window (columns ``date``, ``NDVI_interpolated``).
    season_type : str
        One of ``'two_troughs'``, ``'only_1st_trough'``, ``'only_2nd_trough'``.

    Returns
    -------
    pd.DataFrame
        One-row DataFrame with columns from
        ``['break_pt1', 'break_pt2', 'break_pt3', 'break_pt4']``
        (subset depends on ``season_type``).
    """
    break_point_names = [f"break_pt{n}" for n in range(1, 5)]
    n_segments = 5 if season_type == "two_troughs" else 4

    if season_type == "only_1st_trough":
        break_point_names = break_point_names[:-1]
    elif season_type == "only_2nd_trough":
        break_point_names = break_point_names[1:]

    pwlf_results = piecewise_ndvi_breakpoints(season_ndvi_for_pwlf, n_segments=n_segments)
    break_dates = pd.DataFrame(
        [pd.to_datetime(pwlf_results["break_dates"][1:-1]).normalize()],
        columns=break_point_names,
    )
    return break_dates


def process_single_crop_cycle(
    row: pd.Series,
    df: pd.DataFrame,
    ndvi_threshold_for_approx_season_start: float = 0.3,
    ndvi_threshold_for_approx_season_end: float = 0.3,
    ndvi_threshold_to_extend_data_for_pwlf: float = 0.3,
    season_n_days_buffer: int = 30,
    adaptive_start_threshold: float = None,
    adaptive_end_threshold: float = None,
    season_window_days: int = 90,
    compute_breakpoints: bool = True,
) -> pd.DataFrame:
    """
    Compute milestones for a single crop cycle (one trough-to-trough segment).

    Steps:

    1. Estimate approximate season start and end.
    2. Classify season type (two troughs / only first / only second).
    3. Build the extended NDVI window for piecewise linear fitting.
    4. Detect breakpoints (skipped if ``compute_breakpoints`` is False).
    5. Detect green-up (when a leading trough exists).

    Parameters
    ----------
    row : pd.Series
        One row from ``extract_seasons_trough_to_trough``.
    df : pd.DataFrame
        Daily interpolated NDVI (columns ``date``, ``NDVI_interpolated``).
    ndvi_threshold_for_approx_season_start : float, optional
        Fixed NDVI threshold for season start. Default 0.3.
    ndvi_threshold_for_approx_season_end : float, optional
        Fixed NDVI threshold for season end. Default 0.3.
    ndvi_threshold_to_extend_data_for_pwlf : float, optional
        NDVI threshold for PWLF window extension. Default 0.3.
    season_n_days_buffer : int, optional
        Days added around troughs for PWLF extension. Default 30.
    adaptive_start_threshold : float, optional
        Peak-relative threshold tried first for season start.
    adaptive_end_threshold : float, optional
        Peak-relative threshold tried first for season end.
    season_window_days : int, optional
        Search window when no bounding trough exists. Default 90.
    compute_breakpoints : bool, optional
        Whether to fit the piecewise linear model and populate
        ``break_pt1``…``break_pt4``. The fit is the slowest step in the
        pipeline; set to False to skip it. Default True.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with season metadata, optional breakpoints, and
        optional ``greenup_date``.
    """
    approx = estimate_season_start_end(
        row, df,
        ndvi_threshold_for_approx_season_start,
        ndvi_threshold_for_approx_season_end,
        adaptive_start_threshold=adaptive_start_threshold,
        adaptive_end_threshold=adaptive_end_threshold,
        season_window_days=season_window_days,
    )
    season_milestones = pd.DataFrame([pd.concat([row, approx])])
    row = pd.concat([row, approx])

    if pd.isnull(row.get("trough_after_peak")):
        season_type = "only_1st_trough"
    elif pd.isnull(row.get("trough_before_peak")):
        season_type = "only_2nd_trough"
    else:
        season_type = "two_troughs"

    if compute_breakpoints:
        pwlf_start, pwlf_end = get_dates_for_season_pwlf(
            row, df, season_type,
            ndvi_threshold_to_extend_data_for_pwlf,
            season_n_days_buffer,
        )
        if pd.isnull(pwlf_start) or pd.isnull(pwlf_end):
            return season_milestones

        season_ndvi = df[df["date"].between(pwlf_start, pwlf_end)].copy()
        season_milestones = pd.concat(
            [season_milestones, detect_breakpoints_in_season(season_ndvi, season_type)],
            axis=1,
        )

    if season_type in {"only_1st_trough", "two_troughs"}:
        gu_start, gu_end = get_dates_for_greenup(row, df)
        ndvi_gu = df[df["date"].between(gu_start, gu_end)].copy()
        season_milestones["greenup_date"] = detect_greenup(ndvi_gu)

    return season_milestones


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clip_season_overlaps(df: pd.DataFrame) -> pd.DataFrame:
    for i in range(1, len(df)):
        prev_end = df.at[i - 1, "approx_season_end"]
        curr_start = df.at[i, "approx_season_start"]
        if pd.notna(prev_end) and pd.notna(curr_start) and curr_start <= prev_end:
            df.at[i, "approx_season_start"] = prev_end + pd.Timedelta(days=3)
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def crop_cycle_model(
    df: pd.DataFrame,
    ndvi_threshold_for_approx_season_start: float = 0.36,
    ndvi_threshold_for_approx_season_end: float = 0.4,
    ndvi_threshold_to_extend_data_for_pwlf: float = 0.3,
    season_n_days_buffer: int = 30,
    min_season_length: int = 60,
    min_ndvi_for_peak: float = None,
    ndvi_peak_percentile: float = 90.0,
    ndvi_peak_threshold_ceiling: float = 0.7,
    season_start_pct_of_peak: float = 0.35,
    season_end_pct_of_peak: float = 0.35,
    season_window_days: int = 90,
    compute_breakpoints: bool = True,
    exclude_months: list = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect crop seasons from an NDVI time series for a single polygon.

    High-level steps
    ----------------
    1. Validate and normalise inputs.
    2. Clean NDVI (``clean_ndvi_timeseries``).
    3. Resample to daily frequency and gap-fill with PCHIP interpolation.
    4. Smooth with Whittaker filter (HP filter fallback).
    5. Detect peaks and troughs.
    6. Group into trough-peak-trough seasons.
    7. For each season compute milestones via ``process_single_crop_cycle``.
    8. Deduplicate, filter by minimum length, and renumber seasons.

    Parameters
    ----------
    df : pd.DataFrame
        Input with columns ``date`` (datetime-like) and ``NDVI`` (float).
        Optional column ``poly_name`` (str) names the polygon.
        Must contain data for a **single** polygon.
    ndvi_threshold_for_approx_season_start : float, optional
        NDVI threshold to estimate season start. Default 0.36.
    ndvi_threshold_for_approx_season_end : float, optional
        NDVI threshold to estimate season end. Default 0.4.
    ndvi_threshold_to_extend_data_for_pwlf : float, optional
        NDVI threshold for PWLF window extension. Default 0.3.
    season_n_days_buffer : int, optional
        Temporal buffer (days) around troughs. Default 30.
    min_season_length : int, optional
        Minimum season length (days). Shorter seasons are dropped unless
        their length is NaN (incomplete seasons). Default 60.
    min_ndvi_for_peak : float, optional
        Hard peak NDVI threshold.  If ``None``, an adaptive threshold
        derived from ``ndvi_peak_percentile`` is used.
    ndvi_peak_percentile : float, optional
        Percentile of the smoothed series used as the adaptive peak
        threshold (clipped to ``[0.5, ndvi_peak_threshold_ceiling]``).
        Default 90.
    ndvi_peak_threshold_ceiling : float, optional
        Upper clip for the adaptive peak threshold. Default 0.7.
    season_start_pct_of_peak : float, optional
        Fraction of peak NDVI used as adaptive season-start threshold.
        Default 0.35.
    season_end_pct_of_peak : float, optional
        Fraction of peak NDVI used as adaptive season-end threshold.
        Default 0.35.
    season_window_days : int, optional
        Search window (days) when no bounding trough exists. Default 90.
    compute_breakpoints : bool, optional
        Whether to fit the piecewise linear model and populate
        ``break_pt1``…``break_pt4`` in the report. This fit is the slowest
        step in the pipeline; set to False to skip it and speed up
        processing when breakpoints aren't needed. Default True.
    exclude_months : list, optional
        List of month numbers (1–12) whose peak dates are excluded from the
        output.  ``None`` means no exclusion.

    Returns
    -------
    report : pd.DataFrame
        Per-season milestone table.  Columns:

        - ``season_number``
        - ``poly_name``
        - ``approx_season_start``
        - ``approx_season_end``
        - ``approx_season_length``   (days)
        - ``greenup_date``
        - ``ndvi_peak_date``
        - ``ndvi_peak_value``
        - ``trough_before_peak``
        - ``trough_after_peak``
        - ``troughs_days_distance``
        - ``break_pt1`` … ``break_pt4``   (``None`` if ``compute_breakpoints`` is False)

        Empty if no valid seasons are found.

    ndvi_for_processing : pd.DataFrame
        Daily, interpolated, and smoothed NDVI used internally.
        Columns: ``date``, ``NDVI``, ``NDVI_interpolated``.
    """
    if df.empty:
        logger.info("Input DataFrame is empty.")
        return pd.DataFrame(), pd.DataFrame()

    required = ["date", "NDVI"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logger.info(f"Missing columns: {', '.join(missing)}")
        return pd.DataFrame(), pd.DataFrame()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().all():
        logger.info("date column contains no valid dates.")
        return pd.DataFrame(), pd.DataFrame()

    df["date"] = df["date"].dt.tz_localize(None)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if "poly_name" not in df.columns:
        df["poly_name"] = "anonymous polygon"
    poly_name = df["poly_name"].iloc[0]

    # --- Clean, interpolate, smooth ---
    df_clean = clean_ndvi_timeseries(df)
    ndvi_proc = df_clean.sort_values("date")
    ndvi_proc = ndvi_proc.set_index("date")[["NDVI"]].resample("D").mean()
    ndvi_proc["NDVI"] = ndvi_proc["NDVI"].interpolate(method="pchip")

    try:
        ndvi_proc["NDVI_interpolated"] = whittaker_smooth(ndvi_proc["NDVI"].to_numpy())
    except Exception as e:
        logger.warning(f"Whittaker smoothing failed: {e}. Falling back to HP filter.")
        _, trend = hpfilter(ndvi_proc["NDVI"], lamb=25_000)
        ndvi_proc["NDVI_interpolated"] = trend

    ndvi_proc.reset_index(inplace=True)

    # --- Adaptive peak threshold ---
    if min_ndvi_for_peak is not None:
        effective_peak_threshold = min_ndvi_for_peak
    else:
        p_ndvi = float(np.percentile(ndvi_proc["NDVI_interpolated"], ndvi_peak_percentile))
        effective_peak_threshold = float(np.clip(p_ndvi, 0.5, ndvi_peak_threshold_ceiling))
        logger.info(
            f"Adaptive peak threshold for {poly_name}: {effective_peak_threshold:.3f} "
            f"(p{ndvi_peak_percentile:.0f}={p_ndvi:.3f})"
        )

    p_t = detect_peaks_n_troughs(
        ndvi_proc["NDVI_interpolated"],
        ndvi_proc["date"],
        min_ndvi_for_peak=effective_peak_threshold,
    )

    if p_t.empty or not {"peak", "trough"} <= set(p_t["type"]):
        logger.info(
            f"No valid peaks/troughs for {poly_name} "
            f"({ndvi_proc['date'].min().date()} – {ndvi_proc['date'].max().date()})."
        )
        return pd.DataFrame(), ndvi_proc

    p_t.reset_index(drop=True, inplace=True)
    cropping_cycles = extract_seasons_trough_to_trough(p_t)

    milestones = []
    for _, row in cropping_cycles.iterrows():
        adaptive_start = row["ndvi_peak_value"] * season_start_pct_of_peak
        adaptive_end = row["ndvi_peak_value"] * season_end_pct_of_peak
        milestones.append(process_single_crop_cycle(
            row=row,
            df=ndvi_proc,
            ndvi_threshold_for_approx_season_start=ndvi_threshold_for_approx_season_start,
            ndvi_threshold_for_approx_season_end=ndvi_threshold_for_approx_season_end,
            ndvi_threshold_to_extend_data_for_pwlf=ndvi_threshold_to_extend_data_for_pwlf,
            season_n_days_buffer=season_n_days_buffer,
            adaptive_start_threshold=adaptive_start,
            adaptive_end_threshold=adaptive_end,
            season_window_days=season_window_days,
            compute_breakpoints=compute_breakpoints,
        ))

    report = pd.concat(milestones, ignore_index=True)

    # --- Deduplicate by season start ---
    mask_start = report["approx_season_start"].notna()
    report = pd.concat([
        report[mask_start].sort_values("ndvi_peak_value", ascending=False)
            .drop_duplicates(subset=["approx_season_start"], keep="first"),
        report[~mask_start],
    ], ignore_index=True)

    # --- Deduplicate by season end ---
    mask_end = report["approx_season_end"].notna()
    report = pd.concat([
        report[mask_end].sort_values("ndvi_peak_value", ascending=False)
            .drop_duplicates(subset=["approx_season_end"], keep="first"),
        report[~mask_end],
    ], ignore_index=True)

    def _season_length(d):
        d["approx_season_length"] = (
            d["approx_season_end"] - d["approx_season_start"]
        ).dt.days.astype(float)
        return d

    report = _season_length(report)
    report = report[
        (report["approx_season_length"] >= min_season_length)
        | report["approx_season_length"].isnull()
    ].copy()

    report.sort_values("ndvi_peak_date", inplace=True)
    report["season_number"] = range(1, len(report) + 1)
    report.reset_index(drop=True, inplace=True)
    report = _clip_season_overlaps(report)
    report = _season_length(report)
    report.insert(1, "poly_name", poly_name)

    desired_cols = [
        "season_number", "poly_name",
        "approx_season_start", "approx_season_end", "approx_season_length",
        "greenup_date", "ndvi_peak_date", "ndvi_peak_value",
        "trough_before_peak", "trough_after_peak", "troughs_days_distance",
        "break_pt1", "break_pt2", "break_pt3", "break_pt4",
    ]
    for col in desired_cols:
        if col not in report.columns:
            report[col] = None

    report = report[desired_cols].copy()

    if exclude_months:
        report = report[~report["ndvi_peak_date"].dt.month.isin(exclude_months)].copy()
        report["season_number"] = range(1, len(report) + 1)

    return report, ndvi_proc
