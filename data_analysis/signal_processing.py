"""Reusable transforms for the CapMeter capacitance measurements.

The functions in this module operate on :class:`pandas.DataFrame` objects.
They return a copy of the input dataset, so transforms can be chained without
changing the raw data.
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


CHANNEL_COLUMNS = {
    "left": "CIN2_pF",
    "right": "CIN4_pF",
}
ChannelSelection = str | Sequence[str] | None
Task: TypeAlias = tuple[str, float, float]

DISPLAY_DIRECTIONS = {
    "UD": "up-down",
    "LR": "left-right",
}

def data_load(file_path: str | Path) -> pd.DataFrame:
    """Load a measurement CSV and add continuous time columns.

    The logger can restart its millisecond timestamp counter while appending
    to a file.  ``timestamp_ms`` is therefore made continuous, and
    ``time_seconds`` is measured from the first numeric timestamp.  Rows with
    non-numeric timestamps are retained and receive the preceding timestamp.
    """
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()
    if len(data.columns) < 2:
        raise ValueError("The CSV must contain a timestamp and at least one channel.")

    timestamp_column = data.columns[0]
    raw_timestamps = pd.to_numeric(data[timestamp_column], errors="coerce")
    if raw_timestamps.notna().sum() < 2:
        raise ValueError("The timestamp column must contain at least two numeric values.")

    continuous = pd.Series(index=data.index, dtype="float64")
    offset = 0.0
    last_raw: float | None = None
    last_continuous: float | None = None
    for index, raw_value in raw_timestamps.items():
        if pd.isna(raw_value):
            continuous.loc[index] = last_continuous
            continue
        if last_raw is not None and raw_value < last_raw:
            offset = (last_continuous or 0.0) - raw_value
        current = raw_value + offset
        continuous.loc[index] = current
        last_raw = float(raw_value)
        last_continuous = float(current)

    first_timestamp = continuous.dropna().iloc[0]
    data["timestamp_ms"] = continuous
    data["time_seconds"] = (continuous - first_timestamp) / 1000.0
    for column in data.columns:
        if column not in {timestamp_column, "timestamp_ms", "time_seconds"}:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data

def cut_dataset(
    dataset: pd.DataFrame,
    *,
    start: float = 0,
    end: float = 0,
) -> pd.DataFrame:
    """Remove samples from the start and/or end of a dataset.

    ``start`` and ``end`` are durations in seconds.  The dataset must contain
    ``time_seconds`` or ``timestamp_ms``. The returned dataset is reindexed
    from zero, while the input dataset is unchanged.
    """
    if not np.isfinite(start) or start < 0:
        raise ValueError("start must be a non-negative number of seconds.")
    if not np.isfinite(end) or end < 0:
        raise ValueError("end must be a non-negative number of seconds.")
    if "time_seconds" in dataset:
        time = pd.to_numeric(dataset["time_seconds"], errors="coerce")
    elif "timestamp_ms" in dataset:
        time = pd.to_numeric(dataset["timestamp_ms"], errors="coerce") / 1000
    else:
        raise ValueError("Dataset must contain time_seconds or timestamp_ms.")

    valid_time = time.dropna()
    if valid_time.empty:
        raise ValueError("Dataset contains no valid time values.")
    first_time = valid_time.iloc[0]
    last_time = valid_time.iloc[-1]
    keep = time.ge(first_time + start) & time.le(last_time - end)
    result = dataset.loc[keep].reset_index(drop=True)
    if result.empty:
        raise ValueError("start and end remove the entire dataset.")
    return result

def _channel_columns(dataset: pd.DataFrame, channels: ChannelSelection) -> list[str]:
    """Resolve public channel names and column names used by a dataset."""
    if channels is None or channels == "both":
        requested = list(CHANNEL_COLUMNS)
    elif isinstance(channels, str):
        requested = [channels]
    else:
        requested = list(channels)

    unknown = set(requested) - set(CHANNEL_COLUMNS)
    if unknown:
        raise ValueError(
            f"Unknown channel(s): {', '.join(sorted(unknown))}. "
            f"Use CIN_2, CIN_4, or both."
        )
    columns = [CHANNEL_COLUMNS[channel] for channel in requested]
    missing = [column for column in columns if column not in dataset.columns]
    if missing:
        raise ValueError(f"Dataset is missing channel column(s): {', '.join(missing)}.")
    return columns


def _validate_window(window_size: int) -> None:
    if not isinstance(window_size, (int, np.integer)) or window_size < 1:
        raise ValueError("window_size must be a positive integer.")


def moving_average(
    dataset: pd.DataFrame,
    window_size: int,
    channels: ChannelSelection = "both",
) -> pd.DataFrame:
    """Replace selected channels with their centered moving average."""
    _validate_window(window_size)
    result = dataset.copy()
    for column in _channel_columns(dataset, channels):
        result[column] = dataset[column].rolling(
            window=window_size, min_periods=1, center=True
        ).mean()
    return result


def moving_rms(
    dataset: pd.DataFrame,
    window_size: int,
    channels: ChannelSelection = "both",
) -> pd.DataFrame:
    """Replace selected channels with moving RMS of consecutive differences."""
    _validate_window(window_size)
    result = dataset.copy()
    for column in _channel_columns(dataset, channels):
        result[column] = np.sqrt(
            dataset[column]
            .diff()
            .pow(2)
            .rolling(window=window_size, min_periods=window_size)
            .mean()
        )
    return result


def threshold_transform(
    dataset: pd.DataFrame,
    threshold: float,
    channels: ChannelSelection = "both",
    *,
    inclusive: bool = False,
) -> pd.DataFrame:
    """Convert selected channels to 0/1 values using ``threshold``."""
    if not np.isfinite(threshold):
        raise ValueError("threshold must be a finite number.")
    result = dataset.copy()
    for column in _channel_columns(dataset, channels):
        comparison = (
            dataset[column].ge(threshold)
            if inclusive
            else dataset[column].gt(threshold)
        )
        result[column] = comparison.fillna(False).astype("int8")
    return result


def highpass_transform(
    dataset: pd.DataFrame,
    cutoff_frequency: float,
    sampling_frequency: float | None = None,
    channels: ChannelSelection = "both",
    *,
    filter_order: int = 4,
) -> pd.DataFrame:
    """Apply a zero-phase Butterworth high-pass filter to selected channels."""
    if cutoff_frequency <= 0:
        raise ValueError("cutoff_frequency must be positive.")
    if filter_order < 1:
        raise ValueError("filter_order must be a positive integer.")
    if sampling_frequency is None:
        time = pd.to_numeric(dataset["timestamp_ms"], errors="coerce").dropna()
        intervals = np.diff(time.to_numpy())
        intervals = intervals[intervals > 0]
        if len(intervals) == 0:
            raise ValueError("A positive sampling_frequency is required for this dataset.")
        sampling_frequency = 1000.0 / float(np.median(intervals))
    if sampling_frequency <= 0 or cutoff_frequency >= sampling_frequency / 2:
        raise ValueError("cutoff_frequency must be below the Nyquist frequency.")

    columns = _channel_columns(dataset, channels)
    result = dataset.copy()
    sos = butter(
        filter_order,
        cutoff_frequency / (sampling_frequency / 2),
        btype="highpass",
        output="sos",
    )
    for column in columns:
        values = result[column].interpolate(limit_direction="both").to_numpy()
        pad_length = 3 * (2 * len(sos) + 1)
        if len(values) < pad_length:
            raise ValueError(
                f"Dataset is too short for the selected filter ({pad_length} samples required)."
            )
        result[column] = sosfiltfilt(sos, values)
    return result


def plot(
    dataset: pd.DataFrame,
    channels: ChannelSelection = "both",
    *,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot selected channels and return the figure and axes."""
    columns = _channel_columns(dataset, channels)
    time_column = "time_seconds" if "time_seconds" in dataset else "timestamp_ms"
    if ax is None:
        figure, ax = plt.subplots(figsize=(12, 5))
    else:
        figure = ax.figure
    for column in columns:
        ax.plot(dataset[time_column], dataset[column], label=column)
    ax.set_xlabel("Time (seconds)" if time_column == "time_seconds" else time_column)
    ax.set_ylabel("Capacitance")
    ax.set_title(title or "Capacitance measurements")
    ax.grid(True)
    ax.legend()
    figure.tight_layout()
    return figure, ax


def _build_timeline(
    first_block_tasks: Sequence[Task],
    second_block_tasks: Sequence[Task],
    *,
    baseline_seconds: float,
    rest_seconds: float,
    block_break_seconds: float,
    start_beep_seconds: float,
) -> tuple[pd.DataFrame, float]:
    """Build timeline sections for the classification plot."""
    if any(duration < 0 for _, _, duration in (*first_block_tasks, *second_block_tasks)):
        raise ValueError("Task durations must be non-negative.")
    if min(
        baseline_seconds,
        rest_seconds,
        block_break_seconds,
        start_beep_seconds,
    ) < 0:
        raise ValueError("Timeline durations must be non-negative.")

    sections: list[dict[str, float | str]] = [
        {
            "start": 0.0,
            "end": baseline_seconds,
            "label": "Baseline",
            "kind": "baseline",
        }
    ]
    current = baseline_seconds
    for block_number, tasks in enumerate(
        (first_block_tasks, second_block_tasks),
        start=1,
    ):
        for task_number, (direction, speed_hz, duration) in enumerate(tasks, start=1):
            task_label = (
                f"{block_number}.{task_number}: "
                f"{DISPLAY_DIRECTIONS.get(direction, direction)}, {speed_hz:g} Hz"
            )
            preparation_duration = start_beep_seconds + rest_seconds
            sections.append(
                {
                    "start": current,
                    "end": current + preparation_duration,
                    "label": f"{task_label}\npreparation",
                    "kind": "rest",
                }
            )
            current += preparation_duration
            sections.append(
                {
                    "start": current,
                    "end": current + duration,
                    "label": f"{task_label}\nmovement",
                    "kind": "movement",
                }
            )
            current += duration
        if block_number == 1:
            sections.append(
                {
                    "start": current,
                    "end": current + block_break_seconds,
                    "label": "Block break",
                    "kind": "block_break",
                }
            )
            current += block_break_seconds

    timeline = pd.DataFrame(sections)
    timeline["duration_seconds"] = timeline["end"] - timeline["start"]
    return timeline, current


def plot_capacity_classification(
    dataset: pd.DataFrame,
    movement_classification: pd.DataFrame,
    first_block_tasks: Sequence[Task],
    second_block_tasks: Sequence[Task],
    *,
    baseline_seconds: float = 10,
    rest_seconds: float = 20,
    block_break_seconds: float = 20,
    start_beep_seconds: float = 0.1,
    title: str = "Capacitance and movement classification",
    figsize: tuple[float, float] = (22, 6),
    ylim: tuple[float, float] | None = None,
    offset_seconds: float = 0,
    end_padding_seconds: float = 10,
    channels: ChannelSelection = "both",
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes], pd.DataFrame]:
    """Plot measurements, movement classification, and task timeline.

    ``dataset`` contains the capacitance measurements and
    ``movement_classification`` contains the corresponding binary channel
    values, such as the result of :func:`threshold_transform`. Both datasets
    must have the same number and order of samples.
    """
    raw_columns = _channel_columns(dataset, channels)
    classification_columns = _channel_columns(movement_classification, channels)
    if len(dataset) != len(movement_classification):
        raise ValueError("dataset and movement_classification must have equal length.")

    time_column = "time_seconds" if "time_seconds" in dataset else "timestamp_ms"
    classification_time_column = (
        "time_seconds"
        if "time_seconds" in movement_classification
        else "timestamp_ms"
    )
    timeline, audio_end = _build_timeline(
        first_block_tasks,
        second_block_tasks,
        baseline_seconds=baseline_seconds,
        rest_seconds=rest_seconds,
        block_break_seconds=block_break_seconds,
        start_beep_seconds=start_beep_seconds,
    )
    timeline[["start", "end"]] += offset_seconds
    plot_end = audio_end + offset_seconds + end_padding_seconds
    plot_data = dataset[dataset[time_column] <= plot_end]
    classification_data = movement_classification[
        movement_classification[classification_time_column] <= plot_end
    ]
    if plot_data.empty or classification_data.empty:
        raise ValueError("The datasets contain no measurements before the plot end.")

    colors = {
        "baseline": "#bdbdbd",
        "rest": "#ffe0b2",
        "movement": "#bbdefb",
        "block_break": "#c8e6c9",
    }
    figure, axes = plt.subplots(
        2,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1)},
    )
    raw_axis, classification_axis = axes
    for column in raw_columns:
        raw_axis.plot(plot_data[time_column], plot_data[column], linewidth=0.8, label=column)
    for column in classification_columns:
        classification_axis.step(
            classification_data[classification_time_column],
            classification_data[column],
            where="post",
            linewidth=1.2,
            label=f"{column} movement",
        )

    for section in timeline.to_dict("records"):
        for axis in axes:
            axis.axvspan(
                section["start"],
                section["end"],
                color=colors[section["kind"]],
                alpha=0.28,
                linewidth=0,
            )
        if section["duration_seconds"] >= 5:
            raw_axis.text(
                (section["start"] + section["end"]) / 2,
                raw_axis.get_ylim()[1],
                section["label"],
                ha="center",
                va="top",
                rotation=90,
                fontsize=8,
                clip_on=True,
            )

    if ylim is not None:
        if ylim[0] >= ylim[1]:
            raise ValueError("ylim must contain an increasing (lower, upper) pair.")
        raw_axis.set_ylim(ylim)
    classification_axis.set_yticks([0, 1])
    classification_axis.set_ylim(-0.1, 1.1)
    raw_axis.set_ylabel("Capacitance")
    classification_axis.set_ylabel("Movement")
    classification_axis.set_xlabel("Elapsed time (seconds)")
    raw_axis.set_title(title)
    raw_axis.grid(True, alpha=0.3)
    classification_axis.grid(True, alpha=0.3)
    raw_axis.legend(loc="upper right")
    classification_axis.legend(loc="upper right")
    raw_axis.set_xlim(min(0, offset_seconds), plot_end)
    figure.tight_layout()
    return figure, (raw_axis, classification_axis), timeline