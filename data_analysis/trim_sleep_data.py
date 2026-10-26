"""Trim Apple Watch, Polar, and FDC1004 sleep datasets to one FDC interval."""

from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd


DEFAULT_DATA_DIRECTORY = Path(__file__).parent / "sleep_tests"
DEFAULT_OUTPUT_DIRECTORY = DEFAULT_DATA_DIRECTORY / "datasets_for_training"


def parse_trim_jobs(
    jobs: list[dict[str, object]],
) -> list[tuple[date, float, float]]:
    """Validate notebook-style trim jobs and convert them to typed values."""
    parsed_jobs = []
    for job in jobs:
        recording_date = date.fromisoformat(str(job["date"]))
        front_trim = float(job["front_trim"])
        back_trim = float(job["back_trim"])
        if front_trim < 0 or back_trim < 0:
            raise ValueError(f"Trim values must be non-negative: {job}")
        parsed_jobs.append((recording_date, front_trim, back_trim))
    return parsed_jobs


def _parse_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as error:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD format") from error


def _read_fdc_start(data_directory: Path, recording_date: date) -> datetime:
    path = data_directory / "fdc1004_data" / "data_start.csv"
    starts = pd.read_csv(path)
    starts["date"] = pd.to_datetime(
        starts["date"].astype(str).str.replace(".", "-", regex=False),
        format="%Y-%m-%d",
    ).dt.date
    matching = starts.loc[starts["date"] == recording_date]
    if matching.empty:
        raise ValueError(f"No FDC start time found for {recording_date}: {path}")
    return datetime.combine(
        recording_date,
        datetime.strptime(str(matching.iloc[0]["time"]), "%H:%M:%S").time(),
    )


def _read_fdc_bounds(
    path: Path,
    fdc_start: datetime,
    front_trim: float,
    back_trim: float,
) -> tuple[datetime, datetime, pd.DataFrame]:
    data = pd.read_csv(path)
    if "timestamp_ms" not in data:
        raise ValueError(f"FDC file is missing timestamp_ms: {path}")
    timestamps = pd.to_numeric(data["timestamp_ms"], errors="coerce")
    valid = timestamps.dropna()
    if valid.empty:
        raise ValueError(f"FDC file contains no timestamps: {path}")
    elapsed = timestamps - valid.iloc[0]
    recording_duration = float(elapsed.max()) / 1000
    if front_trim + back_trim >= recording_duration:
        raise ValueError(
            f"Trims remove the complete FDC recording ({recording_duration:.1f} seconds)."
        )
    start = fdc_start + timedelta(seconds=front_trim)
    end = fdc_start + timedelta(seconds=recording_duration - back_trim)
    keep = elapsed.ge(front_trim * 1000) & elapsed.le(
        (recording_duration - back_trim) * 1000
    )
    return start, end, data.loc[keep].copy()


def _trim_apple(
    source: Path,
    destination: Path,
    start: datetime,
    end: datetime,
) -> None:
    data = pd.read_csv(source)
    for column in ("Start time", "End time"):
        data[column] = pd.to_datetime(data[column]).map(
            lambda value: value.tz_localize(None) if value.tzinfo else value
        )
    keep = (data["End time"] > start) & (data["Start time"] < end)
    data = data.loc[keep].copy()
    data["Start time"] = data["Start time"].clip(lower=start, upper=end)
    data["End time"] = data["End time"].clip(lower=start, upper=end)
    if data.empty:
        data = pd.DataFrame(
            [{"Sleep stage": "Awake", "Start time": start, "End time": end}],
            columns=data.columns,
        )
    else:
        data = data.sort_values("Start time").reset_index(drop=True)
        fill_rows = []
        if data.iloc[0]["Start time"] > start:
            fill_rows.append(
                {
                    "Sleep stage": "Awake",
                    "Start time": start,
                    "End time": data.iloc[0]["Start time"],
                }
            )
        if data.iloc[-1]["End time"] < end:
            fill_rows.append(
                {
                    "Sleep stage": "Awake",
                    "Start time": data.iloc[-1]["End time"],
                    "End time": end,
                }
            )
        if fill_rows:
            data = pd.concat([pd.DataFrame(fill_rows), data], ignore_index=True)
    data.to_csv(destination, index=False)


def _polar_start(metadata: list[str]) -> datetime:
    values = next(csv.DictReader(metadata[:2]))
    return datetime.strptime(
        f"{values['Date']} {values['Start time']}",
        "%Y-%m-%d %H:%M:%S",
    )


def _trim_polar(
    source: Path,
    destination: Path,
    start: datetime,
    end: datetime,
) -> None:
    lines = source.read_text(encoding="utf-8-sig").splitlines(keepends=True)
    if len(lines) < 4:
        raise ValueError(f"Polar file is missing metadata or sample rows: {source}")

    metadata = lines[:2]
    sample_header = lines[2]
    fieldnames = next(csv.reader([sample_header.rstrip("\r\n")]))
    sample_rows = list(csv.DictReader(lines[3:], fieldnames=fieldnames))
    polar_start = _polar_start(metadata)
    filtered_rows = []
    for row in sample_rows:
        try:
            sample_time = polar_start + timedelta(
                seconds=pd.to_timedelta(row["Time"]).total_seconds()
            )
        except (KeyError, ValueError):
            continue
        if start <= sample_time < end:
            filtered_rows.append(row)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as output:
        fieldnames = ["Time", "HR (bpm)"]
        writer = csv.DictWriter(
            output,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            {
                "Time": row["Time"],
                "HR (bpm)": row["HR (bpm)"],
            }
            for row in filtered_rows
        )


def trim_sleep_data(
    recording_date: date,
    front_trim: float,
    back_trim: float,
    *,
    data_directory: Path = DEFAULT_DATA_DIRECTORY,
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
) -> None:
    """Trim all date-matched datasets using the FDC recording as the baseline."""
    if front_trim < 0 or back_trim < 0:
        raise ValueError("front_trim and back_trim must be non-negative.")

    fdc_directory = data_directory / "fdc1004_data"
    fdc_source = fdc_directory / (
        f"fdc1004_data_{recording_date.strftime('%Y_%m_%d')}.csv"
    )
    if not fdc_source.is_file():
        raise FileNotFoundError(f"FDC file not found: {fdc_source}")

    fdc_start = _read_fdc_start(data_directory, recording_date)
    start, end, fdc_data = _read_fdc_bounds(
        fdc_source,
        fdc_start,
        front_trim,
        back_trim,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    fdc_data.to_csv(output_directory / fdc_source.name, index=False)

    apple_source = data_directory / "apple_watch_data" / f"{recording_date}.csv"
    if not apple_source.is_file():
        raise FileNotFoundError(f"Apple Watch file not found: {apple_source}")
    _trim_apple(
        apple_source,
        output_directory / apple_source.name,
        start,
        end,
    )

    polar_sources = sorted(
        (data_directory / "Polar_data").glob(
            f"polar_{recording_date.strftime('%Y-%m-%d')}_*.CSV"
        )
    )
    if not polar_sources:
        raise FileNotFoundError(
            f"No Polar file found for {recording_date} in {data_directory / 'Polar_data'}"
        )
    for polar_source in polar_sources:
        _trim_polar(
            polar_source,
            output_directory / polar_source.name,
            start,
            end,
        )

    print(f"Trimmed interval: {start} to {end}")
    print(f"Outputs written to: {output_directory}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("date", type=_parse_date, help="Recording date (YYYY-MM-DD).")
    parser.add_argument(
        "front_trim",
        type=float,
        help="Seconds to remove from the beginning of the FDC recording.",
    )
    parser.add_argument(
        "back_trim",
        type=float,
        help="Seconds to remove from the end of the FDC recording.",
    )
    parser.add_argument("--data-directory", type=Path, default=DEFAULT_DATA_DIRECTORY)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    arguments = parser.parse_args()
    trim_sleep_data(
        arguments.date,
        arguments.front_trim,
        arguments.back_trim,
        data_directory=arguments.data_directory,
        output_directory=arguments.output_directory,
    )


if __name__ == "__main__":
    main()
