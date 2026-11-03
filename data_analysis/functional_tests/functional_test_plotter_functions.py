from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.signal import stft

Task = tuple[str, float, float]
DISPLAY_DIRECTIONS = {
    "UD": "fel-le",
    "LR": "jobbra-balra",
}


def load_measurements(csv_path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """Load a measurement CSV and add time in seconds from its first sample."""
    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip()

    if len(data.columns) < 2:
        raise ValueError("The CSV must contain a timestamp and at least one measurement column.")

    timestamp_column, *capacitance_columns = data.columns
    for column in capacitance_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["timestamp_ms"] = pd.to_numeric(data[timestamp_column], errors="coerce")
    if data["timestamp_ms"].isna().all():
        raise ValueError(f"Timestamp column {timestamp_column!r} contains no numeric values.")

    data["time_seconds"] = (
        data["timestamp_ms"] - data["timestamp_ms"].iloc[0]
    ) / 1000
    return data, capacitance_columns


def build_sections(
    tasks: Sequence[Task],
    start_time: float,
    block_name: str,
    *,
    rest_seconds: float,
    start_beep_seconds: float,
) -> tuple[list[dict[str, float | str]], float]:
    """Build plot sections for one block and return them with its end time."""
    sections: list[dict[str, float | str]] = []
    current = start_time

    for task_number, (direction, speed_hz, duration) in enumerate(tasks, start=1):
        preparation_duration = start_beep_seconds + rest_seconds
        direction_label = DISPLAY_DIRECTIONS.get(direction, direction)
        task_label = f"{block_name}.{task_number}: {direction_label}, {speed_hz:g} Hz"
        sections.append(
            {
                "start": current,
                "end": current + preparation_duration,
                "label": (
                    f"{task_label}\n"
                    "felkészülés"
                ),
                "kind": "rest",
            }
        )
        current += preparation_duration

        sections.append(
            {
                "start": current,
                "end": current + duration,
                "label": (
                    f"{task_label}\n"
                    "mozgás"
                ),
                "kind": "movement",
            }
        )
        current += duration

    return sections, current


def build_timeline(
    first_block_tasks: Sequence[Task],
    second_block_tasks: Sequence[Task],
    *,
    baseline_seconds: float,
    rest_seconds: float,
    block_break_seconds: float,
    start_beep_seconds: float,
) -> tuple[pd.DataFrame, float]:
    """Build the complete timeline used for background sections in the plot."""
    sections: list[dict[str, float | str]] = [
        {
            "start": 0,
            "end": baseline_seconds,
            "label": "Baseline",
            "kind": "baseline",
        }
    ]

    first_sections, first_end = build_sections(
        first_block_tasks,
        baseline_seconds,
        "1",
        rest_seconds=rest_seconds,
        start_beep_seconds=start_beep_seconds,
    )
    sections.extend(first_sections)

    sections.append(
        {
            "start": first_end,
            "end": first_end + block_break_seconds,
            "label": "Blokkok közötti szünet",
            "kind": "block_break",
        }
    )

    second_sections, second_end = build_sections(
        second_block_tasks,
        first_end + block_break_seconds,
        "2",
        rest_seconds=rest_seconds,
        start_beep_seconds=start_beep_seconds,
    )
    sections.extend(second_sections)

    sections_df = pd.DataFrame(sections)
    sections_df["duration_seconds"] = sections_df["end"] - sections_df["start"]
    return sections_df, second_end


def plot_capacity_analysis(
    csv_path: str | Path,
    first_block_tasks: Sequence[Task],
    second_block_tasks: Sequence[Task],
    *,
    baseline_seconds: float = 10,
    rest_seconds: float = 20,
    block_break_seconds: float = 20,
    start_beep_seconds: float = 0.1,
    title: str = "Kapacitás-idő diagram a két audio blokk szakaszaival",
    figsize: tuple[float, float] = (22, 8),
    ylim: tuple[float, float] | None = None,
    offset_seconds: float = 0,
    end_padding_seconds: float = 10,
    rms_window_size: int | None = None,
    rms_threshold_right: float | None = None,
    rms_threshold_left: float | None = None,
    stft_window_seconds: float = 10,
    plot_stft: bool = True,
    crop_start_seconds: float = 0,
    crop_end_seconds: float = 0,
    return_objects: bool = False,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes, plt.Axes], pd.DataFrame] | None:
    """Create capacity, RMSD, and movement-classification plots.

    ``offset_seconds`` shifts the timeline sections relative to the CSV time axis.
    Positive values move them right; negative values move them left.
    ``end_padding_seconds`` controls how long the plot continues after the
    final movement section.
    When ``rms_window_size`` and both channel-specific RMS thresholds are
    provided, the moving-RMSD values and movement classification are plotted
    below the capacity plot.
    ``stft_window_seconds`` controls the duration of each STFT window. The
    capacitance-channel STFTs are plotted in a separate figure when
    ``plot_stft`` is true.
    ``crop_start_seconds`` and ``crop_end_seconds`` crop both figures by that
    many seconds from the left and right side of the time axis. The automatic
    Y-range is fitted to the visible part.
    Set ``return_objects`` to ``True`` to return the figure, axes, and
    timeline table.
    """
    thresholds = (rms_threshold_right, rms_threshold_left)
    if (rms_window_size is None) != (not all(
        threshold is not None for threshold in thresholds
    )):
        raise ValueError(
            "rms_window_size and both channel RMS thresholds must be provided together."
        )
    if rms_window_size is not None and rms_window_size < 1:
        raise ValueError("rms_window_size must be a positive integer.")
    if any(threshold is not None and threshold < 0 for threshold in thresholds):
        raise ValueError("RMS thresholds must be non-negative.")
    if plot_stft and stft_window_seconds <= 0:
        raise ValueError("stft_window_seconds must be positive.")
    if crop_start_seconds < 0 or crop_end_seconds < 0:
        raise ValueError("crop_start_seconds and crop_end_seconds must be non-negative.")

    data, capacitance_columns = load_measurements(csv_path)
    sections_df, audio_end = build_timeline(
        first_block_tasks,
        second_block_tasks,
        baseline_seconds=baseline_seconds,
        rest_seconds=rest_seconds,
        block_break_seconds=block_break_seconds,
        start_beep_seconds=start_beep_seconds,
    )
    sections_df = sections_df.copy()
    sections_df[["start", "end"]] += offset_seconds
    audio_start = offset_seconds
    audio_end += offset_seconds
    plot_end = audio_end + end_padding_seconds
    plot_data = data[data["time_seconds"] <= plot_end]
    if plot_data.empty:
        raise ValueError("The CSV contains no measurements before the plot end.")
    x_min = min(0, audio_start) + crop_start_seconds
    x_max = plot_end - crop_end_seconds
    if x_min >= x_max:
        raise ValueError("crop_start_seconds and crop_end_seconds remove the whole plot.")

    colors = {
        "baseline": "#bdbdbd",
        "rest": "#ffe0b2",
        "movement": "#bbdefb",
        "block_break": "#c8e6c9",
    }
    channel_labels = ("jobb csatorna", "bal csatorna")

    if plot_stft:
        sample_intervals = np.diff(data["time_seconds"].to_numpy())
        sample_intervals = sample_intervals[
            np.isfinite(sample_intervals) & (sample_intervals > 0)
        ]
        if sample_intervals.size == 0:
            raise ValueError("The CSV does not contain enough valid timestamps for an STFT.")
        sampling_frequency = 1 / float(np.median(sample_intervals))
        stft_window_samples = max(2, round(stft_window_seconds * sampling_frequency))
        stft_overlap_samples = stft_window_samples // 2
        stft_figure, stft_axes = plt.subplots(
            len(capacitance_columns),
            1,
            figsize=(22, 4 * len(capacitance_columns)),
            sharex=True,
            squeeze=False,
        )
        color_mesh = None
        for axis, column, channel_label in zip(
            stft_axes[:, 0], capacitance_columns, channel_labels
        ):
            signal = plot_data[column].to_numpy(dtype=float)
            frequencies, times, spectrum = stft(
                signal,
                fs=sampling_frequency,
                nperseg=min(stft_window_samples, len(signal)),
                noverlap=min(stft_overlap_samples, max(0, len(signal) - 1)),
                boundary=None,
            )
            magnitude_db = 20 * np.log10(
                np.maximum(np.abs(spectrum), np.finfo(float).tiny)
            )
            mesh = axis.pcolormesh(
                times + plot_data["time_seconds"].iloc[0] + offset_seconds,
                frequencies,
                magnitude_db,
                shading="auto",
                cmap="magma",
            )
            axis.set_ylabel("Frekvencia (Hz)")
            axis.set_title(f"{channel_label} STFT")
            axis.set_ylim(0, sampling_frequency / 2)
            color_mesh = mesh
            axis.grid(True, alpha=0.2)
        if color_mesh is not None:
            stft_figure.colorbar(
                color_mesh,
                ax=stft_axes[:, 0].tolist(),
                location="top",
                orientation="horizontal",
                pad=0.12,
                label="Magnitúdó (dB)",
            )
        stft_axes[-1, 0].set_xlabel("Eltelt idő (s)")
        for axis in stft_axes[:, 0]:
            axis.set_xlim(x_min, x_max)
        stft_figure.suptitle(
            f"Kapacitásjelek STFT-je ({sampling_frequency:g} Hz mintavétel)",
        )
        stft_figure.tight_layout()

    fig, axes = plt.subplots(
        3,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"height_ratios": (3, 1.5, 1)},
    )
    ax, rmsd_axis, classification_axis = axes

    for i, column in enumerate(capacitance_columns):
        ax.plot(
            plot_data["time_seconds"],
            plot_data[column],
            linewidth=0.8,
            label=channel_labels[i],
        )

    rmsd_lines = []
    threshold_lines = []
    classification_lines = []
    if rms_window_size is not None and all(
        threshold is not None for threshold in thresholds
    ):
        classification_axis.set_ylabel("Mozgásbesorolás")
        classification_axis.set_yticks([0, 1])
        classification_axis.set_ylim(-0.1, 1.1)

        classification_colors = ("tab:purple", "tab:brown")
        for column, color, channel_label, threshold in zip(
            capacitance_columns[:2],
            classification_colors,
            channel_labels,
            thresholds,
        ):
            if threshold is None:
                raise RuntimeError("A channel RMS threshold is unexpectedly missing.")
            rms = np.sqrt(
                data[column]
                .diff()
                .pow(2)
                .rolling(
                    window=rms_window_size,
                    min_periods=rms_window_size,
                )
                .mean()
            )
            threshold_lines.append(
                rmsd_axis.axhline(
                    threshold,
                    color=color,
                    linestyle="--",
                    linewidth=1,
                    label=f"{channel_label} küszöbérték ({threshold:g})",
                )
            )
            rmsd_lines.append(
                rmsd_axis.plot(
                    plot_data["time_seconds"],
                    rms.loc[plot_data.index],
                    color=color,
                    alpha=0.8,
                    linewidth=1.2,
                    label=f"{channel_label} RMSD",
                )[0]
            )
            classification = pd.Series(
                rms > threshold,
                index=data.index,
            ).fillna(False).astype(int)
            line = classification_axis.step(
                plot_data["time_seconds"],
                classification.loc[plot_data.index],
                where="post",
                color=color,
                alpha=0.8,
                linewidth=1.2,
                label=f"{channel_label} mozgásbesorolás",
            )[0]
            classification_lines.append(line)
    else:
        classification_axis.set_ylabel("Mozgásbesorolás")
        classification_axis.set_yticks([0, 1])
        classification_axis.set_ylim(-0.1, 1.1)

    visible_data = plot_data[
        (plot_data["time_seconds"] >= x_min) & (plot_data["time_seconds"] <= x_max)
    ]
    if visible_data.empty:
        visible_data = plot_data
    y_min = visible_data[capacitance_columns].min().min()
    y_max = visible_data[capacitance_columns].max().max()
    y_range = y_max - y_min
    if ylim is None:
        plot_y_min = y_min - 0.05 * y_range
        plot_y_max = y_max + 0.22 * y_range
    else:
        if ylim[0] >= ylim[1]:
            raise ValueError("ylim must contain an increasing (lower, upper) pair.")
        plot_y_min, plot_y_max = ylim

    label_y = plot_y_max - 0.01 * (plot_y_max - plot_y_min)
    for section in sections_df.to_dict("records"):
        for axis in (ax, rmsd_axis, classification_axis):
            axis.axvspan(
                section["start"],
                section["end"],
                color=colors[section["kind"]],
                alpha=0.28,
                linewidth=0,
            )
        if section["duration_seconds"] >= 5:
            ax.text(
                (section["start"] + section["end"]) / 2,
                label_y,
                section["label"],
                ha="center",
                va="top",
                rotation=90,
                fontsize=8,
                clip_on=True,
            )

    rmsd_axis.set_ylabel("RMSDD (pF)")
    classification_axis.set_xlabel("Eltelt idő (s)")
    ax.set_ylabel("Kapacitás (pF)")
    #ax.set_title(title)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(plot_y_min, plot_y_max)
    ax.grid(True, alpha=0.3)
    rmsd_axis.set_xlim(x_min, x_max)
    rmsd_axis.grid(True, alpha=0.3)
    classification_axis.set_xlim(x_min, x_max)
    classification_axis.grid(True, alpha=0.3)
    ax.legend(
        handles=[
            *[
                plt.Line2D([], [], color=line.get_color(), label=line.get_label())
                for line in ax.lines
            ],
            Patch(facecolor=colors["baseline"], alpha=0.5, label="Baseline"),
            Patch(facecolor=colors["rest"], alpha=0.5, label="Szünet"),
            Patch(facecolor=colors["movement"], alpha=0.5, label="Mozgás"),
            Patch(
                facecolor=colors["block_break"],
                alpha=0.5,
                label="Blokkok közötti szünet",
            ),
        ],
        loc="upper right",
    )
    if rmsd_lines:
        rmsd_axis.legend(
            handles=[*rmsd_lines, *threshold_lines],
            loc="upper right",
        )
    if classification_lines:
        classification_axis.legend(handles=classification_lines, loc="upper right")
    fig.tight_layout()
    if return_objects:
        return fig, (ax, rmsd_axis, classification_axis), sections_df
    return None
