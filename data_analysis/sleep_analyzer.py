"""Plot and compare Apple Watch, Polar, and FDC1004 sleep data."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Sequence, TypeAlias

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DateLike: TypeAlias = str | date | datetime | pd.Timestamp

SLEEP_STAGE_NAMES = {
    "Awake": 0,
    "REM": 1,
    "Core": 2,
    "Deep": 3,
}
SLEEP_STAGE_LABELS = ["Awake", "REM", "Core", "Deep"]
SLEEP_STAGE_COLORS = ["#f28e2b", "#f1c40f", "#2878b5", "#12355b"]
PLOT_NAMES = ("sleep_stages", "heart_rate", "capacitance", "rms", "movement")
MOVEMENT_STATES = ("still", "movement", "strong movement")
STRONG_MOVEMENT_THRESHOLD = 0.7


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


def _predict_movement(
    fdc: pd.DataFrame,
    *,
    rms_window_size: int = 20,
    cin2_threshold: float = 0.065,
    cin4_threshold: float = 0.09,
) -> pd.DataFrame:
    if not isinstance(rms_window_size, int) or rms_window_size < 1:
        raise ValueError("rms_window_size must be a positive integer.")
    if cin2_threshold < 0 or cin4_threshold < 0:
        raise ValueError("Channel thresholds must be non-negative.")

    result = fdc.copy()
    rms = {}
    for column in ("CIN2_pF", "CIN4_pF"):
        values = result[column].interpolate(limit_direction="both")
        rms[column] = (
            values.diff()
            .pow(2)
            .rolling(window=rms_window_size, min_periods=rms_window_size)
            .mean()
            .pow(0.5)
        )

    rms = pd.DataFrame(rms, index=result.index)
    result["CIN2_rms"] = rms["CIN2_pF"]
    result["CIN4_rms"] = rms["CIN4_pF"]
    result["CIN2_movement"] = (result["CIN2_rms"] > cin2_threshold).fillna(False)
    result["CIN4_movement"] = (result["CIN4_rms"] > cin4_threshold).fillna(False)
    strong_movement = (
        (result["CIN2_rms"] > STRONG_MOVEMENT_THRESHOLD)
        | (result["CIN4_rms"] > STRONG_MOVEMENT_THRESHOLD)
    ).fillna(False)
    movement = result["CIN2_movement"] | result["CIN4_movement"]
    result["movement_prediction"] = np.select(
        [strong_movement, movement],
        ["strong movement", "movement"],
        default="still",
    )
    return result


def _mark_sleep_stages(fdc: pd.DataFrame, apple: pd.DataFrame) -> pd.DataFrame:
    result = fdc.copy()
    result["sleep_stage"] = pd.Series(pd.NA, index=result.index, dtype="string")
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


def _sleep_stage_statistics(fdc: pd.DataFrame) -> pd.DataFrame:
    values = fdc[["sleep_stage", "movement_prediction"]].dropna()
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
    table = table.rename(
        columns={
            "still": "still_count",
            "movement": "movement_count",
            "strong movement": "strong_movement_count",
        }
    )
    table["phase_vs_sleep_%"] = table.sum(axis=1) / table.to_numpy().sum() * 100
    count_columns = ["still_count", "movement_count", "strong_movement_count"]
    phase_totals = table[count_columns].sum(axis=1)
    table["movement_%"] = table["movement_count"] / phase_totals * 100
    table["strong_movement_%"] = table["strong_movement_count"] / phase_totals * 100
    total_movement = table["movement_count"].sum()
    total_strong_movement = table["strong_movement_count"].sum()
    table["movement_vs_all_movement_%"] = (
        table["movement_count"] / total_movement * 100
        if total_movement
        else np.nan
    )
    table["strong_movement_vs_all_strong_movement_%"] = (
        table["strong_movement_count"] / total_strong_movement * 100
        if total_strong_movement
        else np.nan
    )
    return table


def plot_sleep_data(
    analysis_date: DateLike,
    fdc_start_time: DateLike,
    *,
    data_directory: str | Path | None = None,
    rms_window_size: int = 20,
    cin2_threshold: float = 0.065,
    cin4_threshold: float = 0.09,
    plots: Sequence[str] | None = None,
    show: bool = True,
) -> tuple[plt.Figure, np.ndarray, float]:
    """Plot one date's datasets and return the figure, axes, and REM correlation.

    ``fdc_start_time`` is the absolute local start time of the FDC1004
    recording. The FDC1004 file contains elapsed timestamps only.
    ``plots`` selects panels by name. Supported names are
    ``sleep_stages``, ``heart_rate``, ``capacitance``, ``rms``, and
    ``movement``. By default, all panels are shown.
    """
    selected_plots = tuple(PLOT_NAMES if plots is None else plots)
    unknown_plots = set(selected_plots) - set(PLOT_NAMES)
    if unknown_plots:
        raise ValueError(
            f"Unknown plot(s): {', '.join(sorted(unknown_plots))}. "
            f"Choose from: {', '.join(PLOT_NAMES)}."
        )
    if not selected_plots:
        raise ValueError("plots must contain at least one plot name.")

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
    polar = _load_polar(polar_paths[0])
    fdc = _predict_movement(
        _load_fdc1004(fdc_path, fdc_start_time),
        rms_window_size=rms_window_size,
        cin2_threshold=cin2_threshold,
        cin4_threshold=cin4_threshold,
    )
    fdc = _mark_sleep_stages(fdc, apple)
    correlation = _movement_rem_correlation(fdc)
    print("Sleep-phase movement statistics:")
    print(_sleep_stage_statistics(fdc).to_string(float_format=lambda value: f"{value:.2f}"))

    height_ratios = {
        "sleep_stages": 1.4,
        "heart_rate": 1.2,
        "capacitance": 2.0,
        "rms": 1.5,
        "movement": 1.0,
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
            stage_times = []
            stage_values = []
            for row in apple.itertuples():
                stage_number = SLEEP_STAGE_NAMES[row._1]
                if not stage_times:
                    stage_times.append(row._2)
                    stage_values.append(stage_number)
                else:
                    stage_times.append(row._2)
                    stage_values.append(stage_number)
                axis.axhspan(
                    stage_number - 0.5,
                    stage_number + 0.5,
                    color=SLEEP_STAGE_COLORS[stage_number],
                    alpha=0.35,
                )
            if stage_times:
                stage_times.append(apple.iloc[-1]["End time"])
                stage_values.append(stage_values[-1])
                axis.step(
                    stage_times,
                    stage_values,
                    where="post",
                    color="black",
                    linewidth=1.0,
                )
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
        elif plot_name == "capacitance":
            axis.plot(fdc["timestamp"], fdc["CIN2_pF"], label="CIN2", linewidth=0.8)
            axis.plot(fdc["timestamp"], fdc["CIN4_pF"], label="CIN4", linewidth=0.8)
            axis.set_ylabel("Capacitance (pF)")
            axis.set_title("FDC1004 capacitance")
            axis.legend(loc="upper right")
            axis.grid(True, alpha=0.3)
        elif plot_name == "rms":
            axis.plot(fdc["timestamp"], fdc["CIN2_rms"], label="CIN2 RMS", linewidth=0.8)
            axis.plot(fdc["timestamp"], fdc["CIN4_rms"], label="CIN4 RMS", linewidth=0.8)
            axis.axhline(cin2_threshold, color="tab:blue", linestyle="--", alpha=0.6)
            axis.axhline(cin4_threshold, color="tab:orange", linestyle="--", alpha=0.6)
            axis.set_ylabel("RMS (pF)")
            axis.set_title(f"Moving RMS (window size: {rms_window_size})")
            axis.legend(loc="upper right")
            axis.grid(True, alpha=0.3)
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
