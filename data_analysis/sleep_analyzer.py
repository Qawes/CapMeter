"""Plot and compare Apple Watch, Polar, and FDC1004 sleep data."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Mapping, Sequence, TypeAlias

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import stft


DateLike: TypeAlias = str | date | datetime | pd.Timestamp

SLEEP_STAGE_NAMES = {
    "Awake": 0,
    "REM": 1,
    "Core": 2,
    "Deep": 3,
}
SLEEP_STAGE_LABELS = ["Awake", "REM", "Core", "Deep"]
SLEEP_STAGE_COLORS = ["#f28e2b", "#f1c40f", "#2878b5", "#12355b"]
PLOT_NAMES = (
    "sleep_stages",
    "heart_rate",
    "heart_rate_rmsdd",
    "heart_rate_rmsdd_30s",
    "heart_rate_stft",
    "capacitance",
    # "capacitance_difference_mean",
    "rmsdd",
    "rmsdd_30s",
    "movement",
    # "movement_count",
    "sleep_classification",
)
PLOT_ALIASES = {
    "heart_rate_epoch_rmsdd": "heart_rate_rmsdd_30s",
    "epoch_heart_rate_rmsdd": "heart_rate_rmsdd_30s",
    "epoch_rmsdd": "rmsdd_30s",
    "capacitance_rmsdd_30s": "rmsdd_30s",
    "capacitance_epoch_rmsdd": "rmsdd_30s",
    "fdc_rmsdd_30s": "rmsdd_30s",
    "fdc_epoch_rmsdd": "rmsdd_30s",
}
MOVEMENT_STATES = ("still", "movement", "strong movement")
STRONG_MOVEMENT_THRESHOLD = 0.7
HEART_RATE_RMSDD_MOVEMENT_THRESHOLD = 0.7


def _local_timestamp(value: DateLike) -> pd.Timestamp:
    """Convert a date-like value to a timezone-naive local timestamp."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp


def _date_text(analysis_date: DateLike) -> str:
    return _local_timestamp(analysis_date).strftime("%Y-%m-%d")


def _load_apple_watch(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["Start time"] = pd.to_datetime(data["Start time"]).map(_local_timestamp)
    data["End time"] = pd.to_datetime(data["End time"]).map(_local_timestamp)
    return data.sort_values("Start time").reset_index(drop=True)


def _load_polar(path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(path, nrows=1)
    polar_start = _local_timestamp(
        f"{metadata.loc[0, 'Date']} {metadata.loc[0, 'Start time']}"
    )

    data = pd.read_csv(path, skiprows=2, low_memory=False)
    data["elapsed"] = pd.to_timedelta(data["Time"])
    data["timestamp"] = polar_start + data["elapsed"]
    data["HR (bpm)"] = pd.to_numeric(data["HR (bpm)"], errors="coerce")
    return data


def _load_fdc1004(path: Path, start_time: DateLike) -> pd.DataFrame:
    data = pd.read_csv(path, low_memory=False)
    for column in ("timestamp_ms", "CIN2_pF", "CIN4_pF"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    valid_timestamps = data["timestamp_ms"].dropna()
    if valid_timestamps.empty:
        raise ValueError(f"FDC1004 file contains no timestamps: {path}")

    first_timestamp = valid_timestamps.iloc[0]
    data["elapsed"] = pd.to_timedelta(
        data["timestamp_ms"] - first_timestamp,
        unit="ms",
    )
    data["timestamp"] = _local_timestamp(start_time) + data["elapsed"]
    return data.dropna(subset=["timestamp"]).reset_index(drop=True)


def _add_apple_boundary_awake(
    apple: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Represent missing Apple Watch coverage at the FDC boundaries as Awake."""
    columns = list(apple.columns)
    if apple.empty:
        return pd.DataFrame(
            [{"Sleep stage": "Awake", "Start time": start, "End time": end}],
            columns=columns,
        )

    apple = apple.sort_values("Start time").reset_index(drop=True)
    fill_rows = []
    if apple.iloc[0]["Start time"] > start:
        fill_rows.append(
            {"Sleep stage": "Awake", "Start time": start, "End time": apple.iloc[0]["Start time"]}
        )
    if apple.iloc[-1]["End time"] < end:
        fill_rows.append(
            {"Sleep stage": "Awake", "Start time": apple.iloc[-1]["End time"], "End time": end}
        )
    if fill_rows:
        apple = pd.concat([apple, pd.DataFrame(fill_rows, columns=columns)], ignore_index=True)
        apple = apple.sort_values("Start time").reset_index(drop=True)
    return apple


def _rolling_rmsdd(series: pd.Series, window: int) -> pd.Series:
    """Calculate rolling RMS of successive differences."""
    differences = series.interpolate(limit_direction="both").diff()
    differences = differences.mask(differences.abs() > 4, 0)
    return (
        differences
        .pow(2)
        .rolling(window=window, min_periods=window)
        .mean()
        .pow(0.5)
    )


def _build_30s_epochs(
    apple: pd.DataFrame,
    start_bound: pd.Timestamp | None = None,
    end_bound: pd.Timestamp | None = None,
    epoch_duration_seconds: float = 30.0,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Build constant epoch intervals aligned to Apple Watch start times."""
    apple_sorted = apple.sort_values("Start time").reset_index(drop=True)
    delta = pd.Timedelta(seconds=epoch_duration_seconds)
    if apple_sorted.empty:
        if start_bound is not None and end_bound is not None and start_bound < end_bound:
            cur = start_bound
            epochs = []
            while cur < end_bound:
                nxt = min(end_bound, cur + delta)
                epochs.append((cur, nxt))
                cur = nxt
            return epochs
        return []

    apple_start = apple_sorted["Start time"].iloc[0]
    apple_end = apple_sorted["End time"].iloc[-1]

    leading: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    if start_bound is not None and start_bound < apple_start:
        cur_end = apple_start
        while cur_end > start_bound:
            cur_start = max(start_bound, cur_end - delta)
            leading.append((cur_start, cur_end))
            cur_end = cur_start
        leading.reverse()

    main: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for i, r in enumerate(apple_sorted.itertuples()):
        s = r._2
        e = r._3
        if i > 0:
            prev_e = apple_sorted.iloc[i - 1]["End time"]
            if s > prev_e:
                gap_cur = prev_e
                while gap_cur < s:
                    gap_next = min(s, gap_cur + delta)
                    main.append((gap_cur, gap_next))
                    gap_cur = gap_next
        dur = (e - s).total_seconds()
        n = max(1, int(round(dur / epoch_duration_seconds)))
        for j in range(n):
            t_s = s + pd.Timedelta(seconds=j * epoch_duration_seconds)
            t_e = (
                s + pd.Timedelta(seconds=(j + 1) * epoch_duration_seconds)
                if j < n - 1
                else e
            )
            main.append((t_s, t_e))

    trailing: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    if end_bound is not None and end_bound > apple_end:
        cur_start = apple_end
        while cur_start < end_bound:
            cur_end = min(end_bound, cur_start + delta)
            trailing.append((cur_start, cur_end))
            cur_start = cur_end

    return leading + main + trailing


def _epoch_rmsdd(
    series: pd.Series,
    timestamps: pd.Series,
    epoch_starts: np.ndarray,
    epoch_ends: np.ndarray,
    *,
    max_difference: float | None = 4.0,
    min_samples: int = 2,
) -> np.ndarray:
    """Calculate RMSDD within constant epoch windows."""
    valid_mask = timestamps.notna() & series.notna()
    if not valid_mask.any() or len(epoch_starts) == 0:
        return np.full(len(epoch_starts), np.nan, dtype=float)

    v_timestamps = pd.to_datetime(timestamps[valid_mask]).values
    v_series = series[valid_mask].interpolate(limit_direction="both")
    diffs = v_series.diff()
    if max_difference is not None:
        diffs = diffs.mask(diffs.abs() > max_difference, 0.0)
    sq_diffs = diffs.pow(2).to_numpy()

    s_idx = np.searchsorted(v_timestamps, epoch_starts, side="left")
    e_idx = np.searchsorted(v_timestamps, epoch_ends, side="left")

    result = np.full(len(epoch_starts), np.nan, dtype=float)
    for i in range(len(epoch_starts)):
        si, ei = s_idx[i], e_idx[i]
        if ei - si >= min_samples:
            result[i] = np.sqrt(np.mean(sq_diffs[si:ei]))
    return result


def _map_epochs_to_timestamps(
    timestamps: pd.Series,
    epoch_starts: np.ndarray,
    epoch_ends: np.ndarray,
    epoch_values: np.ndarray,
) -> pd.Series:
    """Map piecewise-constant epoch values onto a sample timestamp series."""
    if len(epoch_starts) == 0 or len(timestamps) == 0:
        return pd.Series(np.nan, index=timestamps.index, dtype=float)

    t_vals = pd.to_datetime(timestamps).values
    idx = np.searchsorted(epoch_ends, t_vals, side="right")
    valid_idx = np.clip(idx, 0, len(epoch_starts) - 1)
    in_bounds = (
        (idx >= 0)
        & (idx < len(epoch_starts))
        & (t_vals >= epoch_starts[valid_idx])
        & (t_vals < epoch_ends[valid_idx])
    )
    mapped = np.where(in_bounds, epoch_values[valid_idx], np.nan)
    return pd.Series(mapped, index=timestamps.index, dtype=float)


def _heart_rate_frequency_bands(
    polar: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Integrate STFT heart-rate amplitude in the LF and HF bands."""
    valid = polar.dropna(subset=["timestamp", "HR (bpm)"]).sort_values("timestamp")
    if len(valid) < 2:
        empty = pd.Series(dtype=float)
        return empty, empty, empty

    timestamps = valid["timestamp"]
    sample_seconds = (timestamps - timestamps.iloc[0]).dt.total_seconds().to_numpy()
    sample_rate = 1.0
    regular_seconds = np.arange(
        sample_seconds[0],
        sample_seconds[-1] + 1 / sample_rate,
        1 / sample_rate,
    )
    heart_rate = np.interp(
        regular_seconds,
        sample_seconds,
        valid["HR (bpm)"].to_numpy(),
    )
    frequencies, times, spectrum = stft(
        heart_rate,
        fs=sample_rate,
        window="hann",
        nperseg=256,
        noverlap=256-32,
        boundary=None,
    )
    amplitude = np.abs(spectrum)
    low_frequency = np.trapz(
        amplitude[(frequencies >= 0.04) & (frequencies < 0.15)],
        frequencies[(frequencies >= 0.04) & (frequencies < 0.15)],
        axis=0,
    )
    high_frequency = np.trapz(
        amplitude[(frequencies >= 0.15) & (frequencies <= 0.4)],
        frequencies[(frequencies >= 0.15) & (frequencies <= 0.4)],
        axis=0,
    )
    plot_timestamps = pd.Series(
        timestamps.iloc[0] + pd.to_timedelta(times, unit="s")
    )
    low_frequency = pd.Series(low_frequency).rolling(
        window=1,
        min_periods=1,
    ).mean()
    high_frequency = pd.Series(high_frequency).rolling(
        window=1,
        min_periods=1,
    ).mean()
    return (
        plot_timestamps,
        low_frequency,
        high_frequency,
    )


# def _rolling_difference_mean(series: pd.Series, window: int) -> pd.Series:
#     """Calculate the rolling mean of absolute successive differences."""
#     differences = series.interpolate(limit_direction="both").diff().abs()
#     differences = differences.mask(differences.abs() > 4, 0)
#     return differences.rolling(window=window, min_periods=window).mean()


def classify_sleep_stages(
    hrv_rmsdd: np.ndarray,
    capacitance_rmsdd: np.ndarray,
    *,
    hrv_threshold_1: float,
    hrv_threshold_2: float,
    c_rmsdd_threshold_1: float,
    c_rmsdd_threshold_2: float,
) -> np.ndarray:
    """Classify epochs into stage numbers (see ``SLEEP_STAGE_NAMES``); NaN if no data.

    - HRV > hrv_threshold_1 and C > c_rmsdd_threshold_1 -> Awake
    - HRV > hrv_threshold_1 and c_rmsdd_threshold_2 < C <= c_rmsdd_threshold_1 -> REM
    - HRV > hrv_threshold_2 (remaining) -> Core
    - HRV <= hrv_threshold_2 -> Deep
    """
    hrv = np.asarray(hrv_rmsdd, dtype=float)
    cap = np.asarray(capacitance_rmsdd, dtype=float)
    stages = np.full(hrv.shape, SLEEP_STAGE_NAMES["Deep"], dtype=float)
    stages[hrv > hrv_threshold_2] = SLEEP_STAGE_NAMES["Core"]
    high_hrv = hrv > hrv_threshold_1
    stages[high_hrv & (cap > c_rmsdd_threshold_2)] = SLEEP_STAGE_NAMES["REM"]
    stages[high_hrv & (cap > c_rmsdd_threshold_1)] = SLEEP_STAGE_NAMES["Awake"]
    stages[np.isnan(hrv) | np.isnan(cap)] = np.nan
    return stages


def _print_classification_comparison(
    predicted: np.ndarray,
    epoch_starts: np.ndarray,
    epoch_ends: np.ndarray,
    apple: pd.DataFrame,
) -> None:
    """Print a confusion matrix and metrics of predicted stages vs Apple Watch."""
    if len(predicted) == 0:
        print("No epochs to compare.")
        return
    mids = (
        pd.to_datetime(epoch_starts)
        + (pd.to_datetime(epoch_ends) - pd.to_datetime(epoch_starts)) / 2
    ).to_numpy()
    apple_sorted = apple.sort_values("Start time").reset_index(drop=True)
    starts = apple_sorted["Start time"].to_numpy()
    ends = apple_sorted["End time"].to_numpy()
    position = np.searchsorted(starts, mids, side="right") - 1
    valid_position = position >= 0
    clipped = np.clip(position, 0, len(apple_sorted) - 1)
    inside = valid_position & (mids < ends[clipped])
    reference = np.full(len(mids), np.nan)
    reference[inside] = (
        apple_sorted["Sleep stage"].map(SLEEP_STAGE_NAMES).to_numpy(dtype=float)[clipped][inside]
    )

    mask = ~np.isnan(predicted) & ~np.isnan(reference)
    if not mask.any():
        print("No overlapping classified epochs to compare.")
        return
    labels = SLEEP_STAGE_LABELS
    y_true = reference[mask].astype(int)
    y_pred = predicted[mask].astype(int)
    n = len(labels)
    matrix = np.zeros((n, n), dtype=int)
    np.add.at(matrix, (y_true, y_pred), 1)
    total = matrix.sum()

    confusion = pd.DataFrame(
        matrix,
        index=pd.Index([f"Apple {l}" for l in labels]),
        columns=pd.Index([f"Pred {l}" for l in labels]),
    )
    true_count = matrix.sum(axis=1)
    pred_count = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = true_positive / pred_count
        recall = true_positive / true_count
        f1 = 2 * precision * recall / (precision + recall)
    per_class = pd.DataFrame(
        {"precision": precision, "recall": recall, "F1": f1, "support": true_count},
        index=labels,
    )
    accuracy = true_positive.sum() / total
    expected = (true_count * pred_count).sum() / total**2
    kappa = (accuracy - expected) / (1 - expected) if expected < 1 else np.nan
    wake_true = y_true == SLEEP_STAGE_NAMES["Awake"]
    wake_pred = y_pred == SLEEP_STAGE_NAMES["Awake"]
    sleep_wake_accuracy = (wake_true == wake_pred).mean()
    sensitivity = (~wake_pred[~wake_true]).mean() if (~wake_true).any() else np.nan
    specificity = wake_pred[wake_true].mean() if wake_true.any() else np.nan

    print(f"\nClassification vs Apple Watch ({total} epochs):")
    print(confusion.to_string())
    print()
    print(per_class.to_string(float_format=lambda v: f"{v:.3f}"))
    print(
        f"\nAccuracy: {accuracy:.3f} | Cohen's kappa: {kappa:.3f} | "
        f"Macro F1: {np.nanmean(f1):.3f}"
    )
    print(
        f"Sleep/wake accuracy: {sleep_wake_accuracy:.3f} | "
        f"Sleep sensitivity: {sensitivity:.3f} | Wake specificity: {specificity:.3f}"
    )


def _predict_movement(
    fdc: pd.DataFrame,
    *,
    rms_window_size: int = 20,
    cin2_threshold: float = 0.065,
    cin4_threshold: float = 0.09,
    heart_rate_rmsdd: pd.Series | None = None,
) -> pd.DataFrame:
    if not isinstance(rms_window_size, int) or rms_window_size < 1:
        raise ValueError("rms_window_size must be a positive integer.")
    if cin2_threshold < 0 or cin4_threshold < 0:
        raise ValueError("Channel thresholds must be non-negative.")
    if heart_rate_rmsdd is not None and not heart_rate_rmsdd.index.equals(fdc.index):
        raise ValueError("heart_rate_rmsdd must use the same index as fdc.")

    result = fdc.copy()
    if heart_rate_rmsdd is not None:
        result["heart_rate_rmsdd"] = heart_rate_rmsdd
    rmsdd = {}
    for column in ("CIN2_pF", "CIN4_pF"):
        rmsdd[column] = _rolling_rmsdd(result[column], rms_window_size)

    rmsdd = pd.DataFrame(rmsdd, index=result.index)
    result["CIN2_rmsdd"] = rmsdd["CIN2_pF"]
    result["CIN4_rmsdd"] = rmsdd["CIN4_pF"]
    result["CIN2_rmsdd"] = result["CIN2_rmsdd"]#.pow(0.5)
    result["CIN4_rmsdd"] = result["CIN4_rmsdd"]#.pow(0.5)
    result["CIN2_movement"] = (result["CIN2_rmsdd"] > cin2_threshold).fillna(False)
    result["CIN4_movement"] = (result["CIN4_rmsdd"] > cin4_threshold).fillna(False)
    strong_movement = (
        (result["CIN2_rmsdd"] > STRONG_MOVEMENT_THRESHOLD)
        | (result["CIN4_rmsdd"] > STRONG_MOVEMENT_THRESHOLD)
    ).fillna(False)
    movement = result["CIN2_movement"] | result["CIN4_movement"]
    if heart_rate_rmsdd is not None:
        low_heart_rate_rmsdd = (
            result["heart_rate_rmsdd"]
            < HEART_RATE_RMSDD_MOVEMENT_THRESHOLD
        ).fillna(False)
        movement = movement & ~low_heart_rate_rmsdd
        strong_movement = strong_movement & ~low_heart_rate_rmsdd
    result["movement_prediction"] = np.select(
        [strong_movement, movement],
        ["strong movement", "movement"],
        default="still",
    )
    return result


def _mark_sleep_stages(fdc: pd.DataFrame, apple: pd.DataFrame) -> pd.DataFrame:
    result = fdc.copy()
    result["sleep_stage"] = pd.Series(pd.NA, index=result.index, dtype="string")
    if not result.empty:
        apple = _add_apple_boundary_awake(
            apple,
            result["timestamp"].min(),
            result["timestamp"].max(),
        )
    for _, sleep_row in apple.iterrows():
        in_stage = result["timestamp"].between(
            sleep_row["Start time"],
            sleep_row["End time"],
            inclusive="left",
        )
        result.loc[in_stage, "sleep_stage"] = sleep_row["Sleep stage"]
    return result


def _movement_rem_correlation(fdc: pd.DataFrame) -> float:
    values = fdc[["movement_prediction", "sleep_stage"]].dropna()
    values = values.assign(rem_phase=(values["sleep_stage"] == "REM").astype(float))
    values["movement_prediction"] = (
        values["movement_prediction"].isin(["movement", "strong movement"]).astype(float)
    )
    if values["movement_prediction"].nunique() < 2:
        return float("nan")
    if values["rem_phase"].nunique() < 2:
        return float("nan")
    return float(values["movement_prediction"].corr(values["rem_phase"]))


def _sleep_stage_statistics(
    fdc: pd.DataFrame,
    polar: pd.DataFrame | None = None,
    stft_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    values = fdc[
        [
            "timestamp",
            "sleep_stage",
            "movement_prediction",
            "CIN2_rmsdd",
            "CIN4_rmsdd",
        ]
    ].dropna(subset=["sleep_stage", "movement_prediction"])
    values = values.rename(columns={"sleep_stage": "phase"})
    table = pd.crosstab(
        values["phase"],
        values["movement_prediction"],
    )
    table = table.reindex(
        index=["Awake", "REM", "Core", "Deep"],
        columns=list(MOVEMENT_STATES),
        fill_value=0,
    )
    table["phase_vs_sleep_%"] = table.sum(axis=1) / table.to_numpy().sum() * 100
    count_columns = list(MOVEMENT_STATES)
    phase_totals = table[count_columns].sum(axis=1)
    table["movement_%"] = table["movement"] / phase_totals * 100
    table["strong_movement_%"] = table["strong movement"] / phase_totals * 100
    table["mean_CIN2_rmsdd"] = values.groupby("phase")["CIN2_rmsdd"].mean().reindex(table.index)
    table["mean_CIN4_rmsdd"] = values.groupby("phase")["CIN4_rmsdd"].mean().reindex(table.index)
    movement_values = values["movement_prediction"].isin(
        ["movement", "strong movement"]
    )
    movement_counts = movement_values.groupby(values["phase"]).sum().reindex(table.index)
    phase_duration = (
        values.groupby("phase")["timestamp"].agg(lambda timestamps: (
            timestamps.max() - timestamps.min()
        ).total_seconds()).reindex(table.index)
    )
    table["average_movement_per_s"] = movement_counts / phase_duration
    total_movement = table["movement"].sum()
    total_strong_movement = table["strong movement"].sum()
    table["movement_vs_all_movement_%"] = (
        table["movement"] / total_movement * 100
        if total_movement
        else np.nan
    )
    table["strong_movement_vs_all_strong_movement_%"] = (
        table["strong movement"] / total_strong_movement * 100
        if total_strong_movement
        else np.nan
    )

    table = table.rename(
        columns={
            "still": "still count",
            "movement": "movement count",
            "strong movement": "strong movement count",
            "phase_vs_sleep_%": "phase/whole sleep in [%]",
            "movement_%": "movement/all points in phase [%]",
            "strong_movement_%": "strong movement/all points [%]",
            "mean_CIN2_rmsdd": "mean C_2 RMSDD [pF]",
            "mean_CIN4_rmsdd": "mean C_4 RMSDD [pF]",
            "average_movement_per_s": "average movement per second [1/s]",
            "movement_vs_all_movement_%": "movement/all movement [%]",
            "strong_movement_vs_all_strong_movement_%": "strong movement/all strong movement [%]",
        }
    )

    # Add means for all signals/plots by sleep phase
    if polar is not None and "sleep_stage" in polar.columns:
        if "HR (bpm)" in polar.columns:
            table["mean HR [bpm]"] = (
                polar.groupby("sleep_stage")["HR (bpm)"].mean().reindex(table.index)
            )
        if "heart_rate_rmsdd" in polar.columns:
            table["mean HR RMSDD [bpm]"] = (
                polar.groupby("sleep_stage")["heart_rate_rmsdd"].mean().reindex(table.index)
            )
        if "heart_rate_rmsdd_30s" in polar.columns:
            table["mean HR 30s RMSDD [bpm]"] = (
                polar.groupby("sleep_stage")["heart_rate_rmsdd_30s"].mean().reindex(table.index)
            )

    if stft_df is not None and "sleep_stage" in stft_df.columns:
        if "low_frequency" in stft_df.columns:
            table["mean LF HRV"] = (
                stft_df.groupby("sleep_stage")["low_frequency"].mean().reindex(table.index)
            )
        if "high_frequency" in stft_df.columns:
            table["mean HF HRV"] = (
                stft_df.groupby("sleep_stage")["high_frequency"].mean().reindex(table.index)
            )

    if "sleep_stage" in fdc.columns:
        if "CIN2_pF" in fdc.columns:
            table["mean CIN2 [pF]"] = (
                fdc.groupby("sleep_stage")["CIN2_pF"].mean().reindex(table.index)
            )
        if "CIN4_pF" in fdc.columns:
            table["mean CIN4 [pF]"] = (
                fdc.groupby("sleep_stage")["CIN4_pF"].mean().reindex(table.index)
            )
        if "CIN2_rmsdd_30s" in fdc.columns:
            table["mean C_2 30s RMSDD [pF]"] = (
                fdc.groupby("sleep_stage")["CIN2_rmsdd_30s"].mean().reindex(table.index)
            )
        if "CIN4_rmsdd_30s" in fdc.columns:
            table["mean C_4 30s RMSDD [pF]"] = (
                fdc.groupby("sleep_stage")["CIN4_rmsdd_30s"].mean().reindex(table.index)
            )

    return table


def plot_sleep_data(
    analysis_date: DateLike,
    fdc_start_time: DateLike,
    *,
    data_directory: str | Path | None = None,
    rmsdd_window_size: int | None = 20,
    heart_rate_rmsdd_window_size: int | None = None,
    capacitance_rmsdd_window_size: int | None = None,
    epoch_step_seconds: float = 30.0,
    epoch_calculation_window_seconds: float | None = None,
    heart_rate_calculation_window_seconds: float | None = None,
    capacitance_calculation_window_seconds: float | None = None,
    epoch_window_seconds: float | None = None,
    epoch_window_alignment: str = "trailing",
    lf_hrv_moving_average_window_size: int | None = None,
    polar_time_offset_seconds: float = 0.0,
    # capacitance_difference_mean_window_size: int | None = None,
    rmsdd_ylim: tuple[float, float] | None = None,
    plot_ylims: Mapping[str, tuple[float, float]] | None = None,
    cin2_threshold: float = 0.065,
    cin4_threshold: float = 0.09,
    hrv_threshold_1: float = 1.0,
    hrv_threshold_2: float = 0.5,
    c_rmsdd_threshold_1: float = 0.09,
    c_rmsdd_threshold_2: float = 0.03,
    plots: Sequence[str] | None = None,
    show: bool = True,
) -> tuple[plt.Figure, np.ndarray, float]:
    """Plot one date's datasets and return the figure, axes, and REM correlation.

    ``fdc_start_time`` is the absolute local start time of the FDC1004
    recording. The FDC1004 file contains elapsed timestamps only.
    ``plots`` selects panels by name. Supported names are
    ``sleep_stages``, ``heart_rate``, ``heart_rate_rmsdd``,
    ``heart_rate_rmsdd_30s``, ``heart_rate_stft``, ``capacitance``,
    ``capacitance_difference_mean``, ``rmsdd``, ``rmsdd_30s``, ``movement``, and
    ``movement_count``. By default, all panels are shown.

    ``rmsdd_window_size`` is retained as a compatibility default for both
    RMSDD calculations. The dedicated window-size arguments override it.
    ``lf_hrv_moving_average_window_size`` enables trailing moving-average
    smoothing of the LF HRV plot when set to a positive integer.
    ``polar_time_offset_seconds`` shifts Polar timestamps before plotting and
    aligning them with the FDC data. Positive values shift them later.
    ``epoch_step_seconds`` sets the stepping interval for constant-window
    epoch evaluation (defaults to 30.0 seconds, aligning to Apple Watch).
    ``epoch_calculation_window_seconds`` sets the duration of the data window
    used to calculate RMSDD for each epoch (defaults to ``epoch_step_seconds``).
    ``heart_rate_calculation_window_seconds`` overrides the calculation window
    for the heart-rate RMSDD only (defaults to
    ``epoch_calculation_window_seconds``); the step is unchanged.
    ``capacitance_calculation_window_seconds`` does the same for the
    capacitance RMSDD only.
    ``epoch_window_alignment`` selects how the calculation window aligns with
    the epoch step ('trailing', 'centered', or 'forward', defaults to 'trailing').
    ``sleep_classification`` classifies each epoch from the 30s HR RMSDD (HRV)
    and the capacitance RMSDD (max of CIN2 and CIN4) using the
    ``hrv_threshold_*`` and ``c_rmsdd_threshold_*`` parameters (see
    ``classify_sleep_stages``). The Polar offset is applied beforehand.
    """
    if heart_rate_rmsdd_window_size is None:
        heart_rate_rmsdd_window_size = rmsdd_window_size
    if capacitance_rmsdd_window_size is None:
        capacitance_rmsdd_window_size = rmsdd_window_size
    # if capacitance_difference_mean_window_size is None:
    #     capacitance_difference_mean_window_size = capacitance_rmsdd_window_size
    if heart_rate_rmsdd_window_size is None or capacitance_rmsdd_window_size is None:
        raise ValueError("At least one RMSDD window size must be provided.")
    if not isinstance(heart_rate_rmsdd_window_size, int) or heart_rate_rmsdd_window_size < 1:
        raise ValueError("heart_rate_rmsdd_window_size must be a positive integer.")
    if not isinstance(capacitance_rmsdd_window_size, int) or capacitance_rmsdd_window_size < 1:
        raise ValueError("capacitance_rmsdd_window_size must be a positive integer.")
    if not isinstance(polar_time_offset_seconds, (int, float)) or isinstance(
        polar_time_offset_seconds, bool
    ):
        raise ValueError("polar_time_offset_seconds must be a number.")
    if (
        lf_hrv_moving_average_window_size is not None
        and (
            not isinstance(lf_hrv_moving_average_window_size, int)
            or lf_hrv_moving_average_window_size < 1
        )
    ):
        raise ValueError(
            "lf_hrv_moving_average_window_size must be a positive integer or None."
        )
    # if (
    #     not isinstance(capacitance_difference_mean_window_size, int)
    #     or capacitance_difference_mean_window_size < 1
    # ):
    #     raise ValueError(
    #         "capacitance_difference_mean_window_size must be a positive integer."
    #     )
    normalized_plot_ylims = {
        PLOT_ALIASES.get(k, k): v for k, v in (plot_ylims or {}).items()
    }
    plot_ylims = normalized_plot_ylims
    if rmsdd_ylim is not None:
        plot_ylims.setdefault("rmsdd", rmsdd_ylim)
        plot_ylims.setdefault("rmsdd_30s", rmsdd_ylim)
    if "heart_rate_rmsdd" in plot_ylims:
        plot_ylims.setdefault("heart_rate_rmsdd_30s", plot_ylims["heart_rate_rmsdd"])
    if "rmsdd" in plot_ylims:
        plot_ylims.setdefault("rmsdd_30s", plot_ylims["rmsdd"])

    fixed_ylim_plots = {"sleep_stages", "movement"}
    configured_fixed_ylim_plots = set(plot_ylims) & fixed_ylim_plots
    if configured_fixed_ylim_plots:
        raise ValueError(
            "Y-limits cannot be configured for: "
            f"{', '.join(sorted(configured_fixed_ylim_plots))}."
        )
    unknown_ylim_plots = set(plot_ylims) - set(PLOT_NAMES)
    if unknown_ylim_plots:
        raise ValueError(
            f"Unknown plot y-limit(s): {', '.join(sorted(unknown_ylim_plots))}."
        )
    selected_plots = tuple(
        PLOT_ALIASES.get(name, name)
        for name in (PLOT_NAMES if plots is None else plots)
    )
    unknown_plots = set(selected_plots) - set(PLOT_NAMES)
    if unknown_plots:
        raise ValueError(
            f"Unknown plot(s): {', '.join(sorted(unknown_plots))}. "
            f"Choose from: {', '.join(PLOT_NAMES)}."
        )
    if not selected_plots:
        raise ValueError("plots must contain at least one plot name.")
    if "sleep_stages" in selected_plots and "sleep_classification" in selected_plots:
        # Keep the classification directly under the Apple Watch stages.
        selected_plots = tuple(
            p for p in selected_plots if p != "sleep_classification"
        )
        index = selected_plots.index("sleep_stages") + 1
        selected_plots = (
            selected_plots[:index] + ("sleep_classification",) + selected_plots[index:]
        )

    date_text = _date_text(analysis_date)
    root = (
        Path(data_directory)
        if data_directory is not None
        else Path(__file__).parent / "sleep_tests"
    )
    apple_path = root / "apple_watch_data" / f"{date_text}.csv"
    fdc_path = root / "fdc1004_data" / f"fdc1004_data_{date_text.replace('-', '_')}.csv"
    polar_paths = sorted((root / "Polar_data").glob(f"polar_{date_text}_*.CSV"))

    if not apple_path.is_file():
        raise FileNotFoundError(f"Apple Watch file not found: {apple_path}")
    if not fdc_path.is_file():
        raise FileNotFoundError(f"FDC1004 file not found: {fdc_path}")
    if not polar_paths:
        raise FileNotFoundError(
            f"No Polar file found for {date_text} in {root / 'Polar_data'}"
        )

    apple = _load_apple_watch(apple_path)
    apple_original = apple.copy()
    polar_parts = []
    heart_rate_band_parts = []
    for polar_path in polar_paths:
        polar_part = _load_polar(polar_path)
        polar_part["timestamp"] += pd.to_timedelta(
            polar_time_offset_seconds,
            unit="s",
        )
        polar_part["heart_rate_rmsdd"] = _rolling_rmsdd(
            polar_part["HR (bpm)"],
            heart_rate_rmsdd_window_size,
        )
        polar_parts.append(polar_part)
        heart_rate_band_parts.append(_heart_rate_frequency_bands(polar_part))
    polar = (
        pd.concat(polar_parts, ignore_index=True)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    raw_fdc = _load_fdc1004(fdc_path, fdc_start_time)
    heart_rate_rmsdd = polar["heart_rate_rmsdd"]
    heart_rate_rmsdd_valid = polar["timestamp"].notna() & heart_rate_rmsdd.notna()
    heart_rate_rmsdd_by_timestamp = pd.Series(
        np.nan,
        index=raw_fdc.index,
        dtype=float,
    )
    if heart_rate_rmsdd_valid.any():
        source_seconds = (
            polar.loc[heart_rate_rmsdd_valid, "timestamp"]
            - polar.loc[heart_rate_rmsdd_valid, "timestamp"].iloc[0]
        ).dt.total_seconds().to_numpy()
        target_seconds = (
            raw_fdc["timestamp"]
            - polar.loc[heart_rate_rmsdd_valid, "timestamp"].iloc[0]
        ).dt.total_seconds().to_numpy()
        source_values = heart_rate_rmsdd.loc[heart_rate_rmsdd_valid].to_numpy()
        in_range = (
            (target_seconds >= source_seconds[0])
            & (target_seconds <= source_seconds[-1])
        )
        heart_rate_rmsdd_by_timestamp.loc[in_range] = np.interp(
            target_seconds[in_range],
            source_seconds,
            source_values,
        )
    fdc = _predict_movement(
        raw_fdc,
        rms_window_size=capacitance_rmsdd_window_size,
        cin2_threshold=cin2_threshold,
        cin4_threshold=cin4_threshold,
        heart_rate_rmsdd=heart_rate_rmsdd_by_timestamp,
    )
    if epoch_window_seconds is not None:
        if epoch_calculation_window_seconds is None:
            epoch_calculation_window_seconds = epoch_window_seconds

    if epoch_step_seconds <= 0:
        raise ValueError("epoch_step_seconds must be positive.")
    if epoch_calculation_window_seconds is None:
        epoch_calculation_window_seconds = epoch_step_seconds
    elif epoch_calculation_window_seconds <= 0:
        raise ValueError("epoch_calculation_window_seconds must be positive.")
    if heart_rate_calculation_window_seconds is None:
        heart_rate_calculation_window_seconds = epoch_calculation_window_seconds
    elif heart_rate_calculation_window_seconds <= 0:
        raise ValueError("heart_rate_calculation_window_seconds must be positive.")
    heart_rate_calculation_window_seconds = float(heart_rate_calculation_window_seconds)
    if capacitance_calculation_window_seconds is None:
        capacitance_calculation_window_seconds = epoch_calculation_window_seconds
    elif capacitance_calculation_window_seconds <= 0:
        raise ValueError("capacitance_calculation_window_seconds must be positive.")
    capacitance_calculation_window_seconds = float(capacitance_calculation_window_seconds)

    valid_alignments = ("trailing", "centered", "forward")
    if epoch_window_alignment not in valid_alignments:
        raise ValueError(
            f"Unknown epoch_window_alignment: {epoch_window_alignment}. "
            f"Choose from: {', '.join(valid_alignments)}."
        )

    all_start_bound = min(raw_fdc["timestamp"].min(), polar["timestamp"].min())
    all_end_bound = max(raw_fdc["timestamp"].max(), polar["timestamp"].max())
    epochs = _build_30s_epochs(
        apple_original,
        start_bound=all_start_bound,
        end_bound=all_end_bound,
        epoch_duration_seconds=epoch_step_seconds,
    )
    epoch_starts = pd.to_datetime([e[0] for e in epochs]).values
    epoch_ends = pd.to_datetime([e[1] for e in epochs]).values

    # Determine calculation window bounds for each epoch
    def _calc_bounds(window_seconds: float):
        calc_delta = pd.to_timedelta(window_seconds, unit="s")
        if window_seconds == epoch_step_seconds:
            return epoch_starts, epoch_ends
        if epoch_window_alignment == "trailing":
            return (
                (pd.to_datetime(epoch_ends) - calc_delta).values,
                pd.to_datetime(epoch_ends).values,
            )
        if epoch_window_alignment == "centered":
            epoch_mid = pd.to_datetime(epoch_starts) + (pd.to_datetime(epoch_ends) - pd.to_datetime(epoch_starts)) / 2
            return (
                (epoch_mid - calc_delta / 2).values,
                (epoch_mid + calc_delta / 2).values,
            )
        return (
            pd.to_datetime(epoch_starts).values,
            (pd.to_datetime(epoch_starts) + calc_delta).values,
        )

    calc_starts, calc_ends = _calc_bounds(capacitance_calculation_window_seconds)
    hr_calc_starts, hr_calc_ends = _calc_bounds(heart_rate_calculation_window_seconds)

    hr_epoch_rmsdd = _epoch_rmsdd(
        polar["HR (bpm)"],
        polar["timestamp"],
        hr_calc_starts,
        hr_calc_ends,
    )
    cin2_epoch_rmsdd = _epoch_rmsdd(
        fdc["CIN2_pF"],
        fdc["timestamp"],
        calc_starts,
        calc_ends,
    )
    cin4_epoch_rmsdd = _epoch_rmsdd(
        fdc["CIN4_pF"],
        fdc["timestamp"],
        calc_starts,
        calc_ends,
    )

    polar["heart_rate_rmsdd_30s"] = _map_epochs_to_timestamps(
        polar["timestamp"],
        epoch_starts,
        epoch_ends,
        hr_epoch_rmsdd,
    )
    fdc["CIN2_rmsdd_30s"] = _map_epochs_to_timestamps(
        fdc["timestamp"],
        epoch_starts,
        epoch_ends,
        cin2_epoch_rmsdd,
    )
    fdc["CIN4_rmsdd_30s"] = _map_epochs_to_timestamps(
        fdc["timestamp"],
        epoch_starts,
        epoch_ends,
        cin4_epoch_rmsdd,
    )

    if epochs:
        epoch_plot_times = pd.to_datetime(
            np.append(epoch_starts, epoch_ends[-1])
        )
        hr_epoch_plot_values = np.append(
            hr_epoch_rmsdd,
            hr_epoch_rmsdd[-1] if len(hr_epoch_rmsdd) else np.nan,
        )
        cin2_epoch_plot_values = np.append(
            cin2_epoch_rmsdd,
            cin2_epoch_rmsdd[-1] if len(cin2_epoch_rmsdd) else np.nan,
        )
        cin4_epoch_plot_values = np.append(
            cin4_epoch_rmsdd,
            cin4_epoch_rmsdd[-1] if len(cin4_epoch_rmsdd) else np.nan,
        )
    else:
        epoch_plot_times = pd.Series([], dtype="datetime64[ns]")
        hr_epoch_plot_values = np.array([], dtype=float)
        cin2_epoch_plot_values = np.array([], dtype=float)
        cin4_epoch_plot_values = np.array([], dtype=float)

    apple = _add_apple_boundary_awake(
        apple,
        fdc["timestamp"].min(),
        fdc["timestamp"].max(),
    )
    fdc = _mark_sleep_stages(fdc, apple)
    heart_rate_band_times = pd.concat(
        [part[0] for part in heart_rate_band_parts],
        ignore_index=True,
    )
    low_frequency = pd.concat(
        [part[1] for part in heart_rate_band_parts],
        ignore_index=True,
    )
    high_frequency = pd.concat(
        [part[2] for part in heart_rate_band_parts],
        ignore_index=True,
    )
    if lf_hrv_moving_average_window_size is not None:
        low_frequency = low_frequency.rolling(
            window=lf_hrv_moving_average_window_size,
            min_periods=1,
        ).mean()
    polar_marked = _mark_sleep_stages(polar, apple)
    stft_df = pd.DataFrame(
        {
            "timestamp": heart_rate_band_times,
            "low_frequency": low_frequency,
            "high_frequency": high_frequency,
        }
    )
    stft_df = _mark_sleep_stages(stft_df, apple)

    correlation = _movement_rem_correlation(fdc)
    print("Sleep-phase movement statistics:")
    statistics = _sleep_stage_statistics(fdc, polar=polar_marked, stft_df=stft_df)
    print(
        statistics.to_string(
            float_format=lambda value: f"{value:.2f}",
            formatters={
                "mean_CIN2_rmsdd": lambda value: f"{value:.4f}",
                "mean_CIN4_rmsdd": lambda value: f"{value:.4f}",
                "mean C_2 RMSDD [pF]": lambda value: f"{value:.4f}",
                "mean C_4 RMSDD [pF]": lambda value: f"{value:.4f}",
                "average_movement_per_s": lambda value: f"{value:.4f}",
                "average movement per second [1/s]": lambda value: f"{value:.4f}",
                "mean HR [bpm]": lambda value: f"{value:.2f}",
                "mean HR RMSDD [bpm]": lambda value: f"{value:.4f}",
                "mean HR 30s RMSDD [bpm]": lambda value: f"{value:.4f}",
                "mean LF HRV": lambda value: f"{value:.4f}",
                "mean HF HRV": lambda value: f"{value:.4f}",
                "mean CIN2 [pF]": lambda value: f"{value:.2f}",
                "mean CIN4 [pF]": lambda value: f"{value:.2f}",
                "mean C_2 30s RMSDD [pF]": lambda value: f"{value:.4f}",
                "mean C_4 30s RMSDD [pF]": lambda value: f"{value:.4f}",
            },
        )
    )

    classified = classify_sleep_stages(
        hr_epoch_rmsdd,
        np.fmax(cin2_epoch_rmsdd, cin4_epoch_rmsdd),
        hrv_threshold_1=hrv_threshold_1,
        hrv_threshold_2=hrv_threshold_2,
        c_rmsdd_threshold_1=c_rmsdd_threshold_1,
        c_rmsdd_threshold_2=c_rmsdd_threshold_2,
    )
    if "sleep_classification" in selected_plots:
        _print_classification_comparison(classified, epoch_starts, epoch_ends, apple)

    height_ratios = {
        "sleep_stages": 1.4,
        "heart_rate": 1.2,
        "heart_rate_rmsdd": 1.2,
        "heart_rate_rmsdd_30s": 1.2,
        "heart_rate_stft": 1.2,
        "capacitance": 2.0,
        # "capacitance_difference_mean": 1.5,
        "rmsdd": 1.5,
        "rmsdd_30s": 1.5,
        "movement": 1.0,
        # "movement_count": 1.0,
        "sleep_classification": 1.4,
    }
    figure, axes = plt.subplots(
        len(selected_plots),
        1,
        figsize=(16, 3 * len(selected_plots)),
        sharex=True,
        gridspec_kw={
            "height_ratios": [height_ratios[name] for name in selected_plots]
        },
    )
    axes = np.atleast_1d(axes)

    for axis, plot_name in zip(axes, selected_plots):
        if plot_name == "sleep_stages":
            previous_stage_number = None
            for row in apple.itertuples():
                stage_number = SLEEP_STAGE_NAMES[row._1]
                axis.axhspan(
                    stage_number - 0.5,
                    stage_number + 0.5,
                    color=SLEEP_STAGE_COLORS[stage_number],
                    alpha=0.35,
                )
                if previous_stage_number is not None:
                    axis.plot(
                        [row._2, row._2],
                        [previous_stage_number, stage_number],
                        color="black",
                        linewidth=1.0,
                    )
                axis.plot(
                    [row._2, row._3],
                    [stage_number, stage_number],
                    color="black",
                    linewidth=1.0,
                )
                previous_stage_number = stage_number
            axis.set_ylabel("Sleep stage")
            axis.set_title(f"Sleep stages ({date_text})")
            axis.set_yticks(range(len(SLEEP_STAGE_LABELS)))
            axis.set_yticklabels(SLEEP_STAGE_LABELS)
            axis.set_ylim(len(SLEEP_STAGE_LABELS) - 0.5, -0.5)
            axis.grid(True, axis="x", alpha=0.3)
        elif plot_name == "heart_rate":
            polar_valid = polar.dropna(subset=["HR (bpm)"])
            axis.plot(
                polar_valid["timestamp"],
                polar_valid["HR (bpm)"],
                color="tab:red",
                linewidth=1,
                label="Heart rate",
            )
            axis.set_ylabel("HR (bpm)")
            axis.set_title("Polar heart rate")
            axis.legend(loc="upper right")
            axis.grid(True, alpha=0.3)
        elif plot_name == "heart_rate_rmsdd":
            axis.plot(
                polar["timestamp"],
                heart_rate_rmsdd,
                color="tab:purple",
                linewidth=1,
                label="Heart-rate RMSDD",
            )
            axis.set_ylabel("HR RMSDD (bpm)")
            axis.set_title(
                f"Heart-rate RMSDD (window size: {heart_rate_rmsdd_window_size})"
            )
            axis.legend(loc="upper right")
            axis.grid(True, alpha=0.3)
        elif plot_name == "heart_rate_rmsdd_30s":
            hr_w = heart_rate_calculation_window_seconds
            calc_w_str = f"{int(hr_w) if hr_w.is_integer() else hr_w}s"
            axis.step(
                epoch_plot_times,
                hr_epoch_plot_values,
                where="post",
                color="tab:purple",
                linewidth=1.0,
                label=f"Heart-rate RMSDD ({calc_w_str} window)",
            )
            axis.set_ylabel("HR RMSDD (bpm)")
            step_str = f"{int(epoch_step_seconds) if epoch_step_seconds.is_integer() else epoch_step_seconds}s"
            if hr_w == epoch_step_seconds:
                axis.set_title(f"Heart-rate RMSDD (constant {calc_w_str} window)")
            else:
                axis.set_title(f"Heart-rate RMSDD ({calc_w_str} window, {step_str} step)")
            axis.legend(loc="upper right")
            axis.grid(True, alpha=0.3)
        elif plot_name == "heart_rate_stft":
            axis.plot(
                heart_rate_band_times,
                low_frequency,
                color="tab:blue",
                linewidth=1.0,
                label="LF amplitude (0.04-0.15 Hz)",
            )
            axis.plot(
                heart_rate_band_times,
                high_frequency,
                color="tab:orange",
                linewidth=1.0,
                label="HF amplitude (0.15-0.4 Hz)",
            )
            axis.set_ylabel("Integrated amplitude")
            axis.set_title("HRV calculated with STFT")
            axis.legend(loc="upper right")
            axis.grid(True, alpha=0.3)
        elif plot_name == "capacitance":
            axis.plot(fdc["timestamp"], fdc["CIN2_pF"], label="CIN2", linewidth=0.8)
            axis.plot(fdc["timestamp"], fdc["CIN4_pF"], label="CIN4", linewidth=0.8)
            axis.set_ylabel("Capacitance (pF)")
            axis.set_title("FDC1004 capacitance")
            axis.legend(loc="upper right")
            axis.grid(True, alpha=0.3)
        # elif plot_name == "capacitance_difference_mean":
        #     cin2_difference_mean = _rolling_difference_mean(
        #         fdc["CIN2_pF"],
        #         capacitance_difference_mean_window_size,
        #     )
        #     cin4_difference_mean = _rolling_difference_mean(
        #         fdc["CIN4_pF"],
        #         capacitance_difference_mean_window_size,
        #     )
        #     axis.plot(
        #         fdc["timestamp"],
        #         cin2_difference_mean,
        #         label="CIN2 difference mean",
        #         linewidth=0.8,
        #     )
        #     axis.plot(
        #         fdc["timestamp"],
        #         cin4_difference_mean,
        #         label="CIN4 difference mean",
        #         linewidth=0.8,
        #     )
        #     axis.set_ylabel("Mean difference (pF)")
        #     axis.set_title(
        #         "Capacitance difference rolling mean "
        #         f"(window size: {capacitance_difference_mean_window_size})"
        #     )
        #     axis.legend(loc="upper right")
        #     axis.grid(True, alpha=0.3)
        elif plot_name == "rmsdd":
            axis.plot(fdc["timestamp"], fdc["CIN2_rmsdd"], label="CIN2 RMSDD", linewidth=0.8)
            axis.plot(fdc["timestamp"], fdc["CIN4_rmsdd"], label="CIN4 RMSDD", linewidth=0.8)
            axis.axhline(cin2_threshold, color="tab:blue", linestyle="--", alpha=0.6)
            axis.axhline(cin4_threshold, color="tab:orange", linestyle="--", alpha=0.6)
            axis.set_ylabel("RMSDD (pF)")
            axis.set_title(
                f"RMSDD (window size: {capacitance_rmsdd_window_size})"
            )
            axis.legend(loc="upper right")
            axis.grid(True, alpha=0.3)
        elif plot_name == "rmsdd_30s":
            cap_w = capacitance_calculation_window_seconds
            calc_w_str = f"{int(cap_w) if cap_w.is_integer() else cap_w}s"
            step_str = f"{int(epoch_step_seconds) if epoch_step_seconds.is_integer() else epoch_step_seconds}s"
            axis.step(
                epoch_plot_times,
                cin2_epoch_plot_values,
                where="post",
                label=f"CIN2 RMSDD ({calc_w_str})",
                linewidth=0.8,
            )
            axis.step(
                epoch_plot_times,
                cin4_epoch_plot_values,
                where="post",
                label=f"CIN4 RMSDD ({calc_w_str})",
                linewidth=0.8,
            )
            axis.axhline(cin2_threshold, color="tab:blue", linestyle="--", alpha=0.6)
            axis.axhline(cin4_threshold, color="tab:orange", linestyle="--", alpha=0.6)
            axis.set_ylabel("RMSDD (pF)")
            if cap_w == epoch_step_seconds:
                axis.set_title(f"RMSDD (constant {calc_w_str} window)")
            else:
                axis.set_title(f"RMSDD ({calc_w_str} window, {step_str} step)")
            axis.legend(loc="upper right")
            axis.grid(True, alpha=0.3)
        elif plot_name == "sleep_classification":
            if len(classified):
                class_values = np.append(classified, classified[-1])
                for stage_number in range(len(SLEEP_STAGE_LABELS)):
                    axis.axhspan(
                        stage_number - 0.5,
                        stage_number + 0.5,
                        color=SLEEP_STAGE_COLORS[stage_number],
                        alpha=0.35,
                    )
                axis.step(
                    epoch_plot_times,
                    class_values,
                    where="post",
                    color="black",
                    linewidth=1.0,
                )
            axis.set_ylabel("Sleep stage")
            axis.set_title(
                "Classified sleep stages "
                f"(HRV thr: {hrv_threshold_1}/{hrv_threshold_2}, "
                f"C RMSDD thr: {c_rmsdd_threshold_1}/{c_rmsdd_threshold_2})"
            )
            axis.set_yticks(range(len(SLEEP_STAGE_LABELS)))
            axis.set_yticklabels(SLEEP_STAGE_LABELS)
            axis.set_ylim(len(SLEEP_STAGE_LABELS) - 0.5, -0.5)
            axis.grid(True, axis="x", alpha=0.3)
        elif plot_name == "movement":
            movement_levels = fdc["movement_prediction"].map(
                {state: index for index, state in enumerate(MOVEMENT_STATES)}
            )
            axis.step(
                fdc["timestamp"],
                movement_levels,
                where="post",
                color="tab:green",
                linewidth=1.2,
                label="Predicted movement",
            )
            axis.fill_between(
                fdc["timestamp"],
                0,
                movement_levels,
                step="post",
                color="tab:green",
                alpha=0.25,
            )
            axis.set_ylabel("Movement")
            axis.set_yticks(range(len(MOVEMENT_STATES)))
            axis.set_yticklabels(MOVEMENT_STATES)
            axis.set_ylim(-0.1, len(MOVEMENT_STATES) - 0.9)
            axis.set_title(
                f"Movement prediction (REM correlation: {correlation:.4f})"
                if np.isfinite(correlation)
                else "Movement prediction (REM correlation: undefined)"
            )
            axis.legend(loc="upper right")
            axis.grid(True, alpha=0.3)
        # elif plot_name == "movement_count":
        #     movement_count = fdc["movement_prediction"].isin(
        #         ["movement", "strong movement"]
        #     ).rolling(window=20, min_periods=1).sum()
        #     axis.plot(
        #         fdc["timestamp"],
        #         movement_count,
        #         color="tab:green",
        #         linewidth=1.2,
        #         label="Movement count (last 20)",
        #     )
        #     axis.set_ylabel("Movement count")
        #     axis.set_title("Movement count over last 20 measurements")
        #     axis.legend(loc="upper right")
        #     axis.grid(True, alpha=0.3)

        if plot_name in plot_ylims:
            axis.set_ylim(*plot_ylims[plot_name])

    all_times = pd.concat(
        [
            apple["Start time"],
            apple["End time"],
            polar["timestamp"],
            fdc["timestamp"],
        ]
    ).dropna()
    axes[-1].set_xlim(all_times.min(), all_times.max())
    axes[-1].set_xlabel("Local time")
    axes[-1].xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    figure.autofmt_xdate()
    figure.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.08,
        top=0.90,
        hspace=0.35,
    )

    if show:
        plt.show()
    else:
        plt.close(figure)
    return figure, axes, correlation


if __name__ == "__main__":
    _, _, movement_rem_correlation = plot_sleep_data(
        "2026-10-16",
        "2026-10-16 01:07:07",
    )
    print(f"Movement/REM Pearson correlation: {movement_rem_correlation:.4f}")
