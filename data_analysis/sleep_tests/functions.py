from dataclasses import dataclass

import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, sosfreqz
import numpy as np



@dataclass
class MeasurementData:
    time_hours: pd.Series
    time_min: pd.Series
    capacitance_1: pd.Series
    capacitance_2: pd.Series
    sampling_frequency: float
    capacitance_1_key: str
    capacitance_2_key: str


def load_data(file_path) -> MeasurementData:
    
    data = pd.read_csv(file_path)
    data.columns = data.columns.str.strip()

    # Get column names from the CSV header row
    timestamp_key, capacitance_1_key, capacitance_2_key = data.columns[:3]
    time = data[timestamp_key]
    
    
    # Convert timestamps while allowing session-marker rows
    time_numeric = pd.to_numeric(time, errors="coerce")

    # Build a continuous timestamp across restarted logging sessions
    continuous_time_ms = pd.Series(index=time.index, dtype="float64")
    offset_ms = 0.0
    last_raw_time = None
    last_continuous_time = None

    for index, raw_value in time_numeric.items():
        if pd.isna(raw_value):
            # Keep an append-session marker in the table without breaking the plot
            continuous_time_ms.loc[index] = last_continuous_time
            continue

        if last_raw_time is not None and raw_value < last_raw_time:
            # Timestamp counter restarted
            offset_ms = last_continuous_time - raw_value

        continuous_value = raw_value + offset_ms
        continuous_time_ms.loc[index] = continuous_value

        last_raw_time = raw_value
        last_continuous_time = continuous_value

    time_hours = (continuous_time_ms - continuous_time_ms.iloc[0]) / 3600000
    time_min = (continuous_time_ms - continuous_time_ms.iloc[0]) / 60000
    sampling_frequency = 1000 / np.median(np.diff(continuous_time_ms.dropna().to_numpy()))
        

    capacitance_1 = pd.to_numeric(data[capacitance_1_key], errors="coerce")
    capacitance_2 = pd.to_numeric(data[capacitance_2_key], errors="coerce")
    
    return MeasurementData(
        time_hours=time_hours,
        time_min=time_min,
        capacitance_1=capacitance_1,
        capacitance_2=capacitance_2,
        sampling_frequency=sampling_frequency,
        capacitance_1_key=capacitance_1_key,
        capacitance_2_key=capacitance_2_key,
    )

def plot_raw_data(measurement: MeasurementData):


    # Capacitance 1
    plt.figure(figsize=(10, 5))
    plt.plot(measurement.time_hours, measurement.capacitance_1)
    plt.xlabel("Elapsed time (hours)")
    plt.ylabel("Capacitance 1 (pF)")
    plt.title("Capacitance 1 vs Time")
    plt.grid(True)
    plt.show()

    # Capacitance 2
    plt.figure(figsize=(10, 5))
    plt.plot(measurement.time_hours, measurement.capacitance_2, color="orange")
    plt.xlabel("Elapsed time (hours)")
    plt.ylabel("Capacitance 2 (pF)")
    plt.title("Capacitance 2 vs Time")
    plt.grid(True)
    plt.show()

    # Both capacitances
    plt.figure(figsize=(10, 5))
    plt.plot(
        measurement.time_hours,
        measurement.capacitance_1,
        label="Capacitance 1",
    )
    plt.plot(
        measurement.time_hours,
        measurement.capacitance_2,
        label="Capacitance 2",
    )
    plt.xlabel("Elapsed time (hours)")
    plt.ylabel("Capacitance (pF)")
    plt.title("Capacitances vs Time")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
def plot_filtered_data(measurement: MeasurementData, cutoff_frequency):

    filter_order = 4
    nyquist_frequency = measurement.sampling_frequency / 2

    # Butterworth felüláteresztő szűrő
    sos = butter(
        filter_order,
        cutoff_frequency / nyquist_frequency,
        btype="highpass",
        output="sos",
    )

    capacitance_1_original = (
        pd.to_numeric(measurement.capacitance_1, errors="coerce")
        .interpolate(limit_direction="both")
        .to_numpy()
    )

    capacitance_2_original = (
        pd.to_numeric(measurement.capacitance_2, errors="coerce")
        .interpolate(limit_direction="both")
        .to_numpy()
    )

    capacitance_1_highpass = sosfiltfilt(sos, capacitance_1_original)
    capacitance_2_highpass = sosfiltfilt(sos, capacitance_2_original)

    # Szűrt jelek ábrázolása
    fig_highpass, axes_highpass = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes_highpass[0].plot(
        measurement.time_min,
        capacitance_1_highpass,
        color="tab:blue",
    )
    axes_highpass[0].set_ylabel("Capacitance (pF)")
    axes_highpass[0].set_title(
        f"{measurement.capacitance_1_key} – felüláteresztő szűrt jel"
    )
    axes_highpass[0].grid(True)

    axes_highpass[1].plot(
        measurement.time_min,
        capacitance_2_highpass,
        color="tab:orange",
    )
    axes_highpass[1].set_xlabel("Eltelt idő (perc)")
    axes_highpass[1].set_ylabel("Capacitance (pF)")
    axes_highpass[1].set_title(
        f"{measurement.capacitance_2_key} – felüláteresztő szűrt jel"
    )
    axes_highpass[1].grid(True)

    plt.tight_layout()
    plt.show()
    
    return sos, capacitance_1_highpass, capacitance_2_highpass  # Return the filter coefficients for Bode plot
    
    
def plot_bode_diag(cutoff_frequency, sampling_frequency, sos):
    # Frequency response of the Butterworth high-pass filter
    frequencies_bode, response = sosfreqz(
        sos,
        worN=4096,
        fs=int(sampling_frequency),
    )

    magnitude_db = 20 * np.log10(np.maximum(np.abs(response), 1e-12))
    phase_degrees = np.unwrap(np.angle(response)) * 180 / np.pi

    # Exclude zero frequency for logarithmic plotting
    positive = frequencies_bode > 0

    fig_bode, axes_bode = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axes_bode[0].semilogx(
        frequencies_bode[positive],
        magnitude_db[positive],
        color="tab:blue",
    )
    axes_bode[0].axvline(
        cutoff_frequency,
        color="red",
        linestyle="--",
        label=f"Cutoff: {cutoff_frequency} Hz",
    )
    axes_bode[0].set_ylabel("Magnitude (dB)")
    axes_bode[0].set_title("Bode Diagram of the High-Pass Butterworth Filter")
    axes_bode[0].grid(True, which="both")
    axes_bode[0].legend()

    axes_bode[1].semilogx(
        frequencies_bode[positive],
        phase_degrees[positive],
        color="tab:orange",
    )
    axes_bode[1].set_xlabel("Frequency (Hz)")
    axes_bode[1].set_ylabel("Phase (degrees)")
    axes_bode[1].grid(True, which="both")

    plt.tight_layout()
    plt.show()