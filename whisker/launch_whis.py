import os
import subprocess
import sys

import numpy as np
import pandas as pd
import streamlit as st
from trace_colors import *

app_path = os.path.join(
    os.path.dirname(__file__), "..", "sleepscoring", "sleepscore_main.py"
)

st.markdown("# Format and View Texture Disrimination Experiment")
st.markdown(
    "Data Directorty can be located anywhere and should contain: events.csv, frames.csv, vid.mp4"
)


# DEFINE THESE IF FORMATTING IS NEEDED
data_dir = st.text_input("Data Directory", value="/data/exp_1")

events_name = st.text_input("Events File Name", value="events.csv")
frames_name = st.text_input("Frames File Name", value="frames.csv")
video_name = st.text_input("Video File Name", value="vid.mp4")

time_col_events = st.text_input("Time Column in Events File", value="Pi5_gTime_s")
time_col_frames = st.text_input("Time Column in Frames File", value="pi5_sync_s")


if st.button("Format Data"):
    ev = pd.read_csv(os.path.join(data_dir, events_name))
    ev.rename(columns={time_col_events: "time"}, inplace=True)
    array_start = 0
    array_end = round(ev["time"].values.max())
    fs = 1000

    master_times_array = np.arange(array_start, array_end, 1 / fs)

    # First we save the obvious discrete events: licks
    lick_times = ev.loc[ev.Lick_Detected == "Lick"]["time"].values
    lick_times_y = np.ones_like(lick_times)
    np.save(os.path.join(data_dir, "lick_events_t.npy"), lick_times)
    np.save(os.path.join(data_dir, "lick_events_y.npy"), lick_times_y)

    # ... and valve opens/closes
    valve_opens = ev.loc[ev.Valve_timing == "Open"]["time"].values
    valve_closes = ev.loc[ev.Valve_timing == "Close"]["time"].values
    valve_opens_y = np.ones_like(valve_opens) * 2
    valve_closes_y = np.ones_like(valve_closes)
    # (interleave so they are displayed on a single scatter)
    valve_times_full = np.array(list(zip(valve_opens, valve_closes))).flatten()
    valve_vals_full = np.array(list(zip(valve_opens_y, valve_closes_y))).flatten()
    np.save(os.path.join(data_dir, "valve_events_t.npy"), valve_times_full)
    np.save(os.path.join(data_dir, "valve_events_y.npy"), valve_vals_full)

    # Now we format the motor events into a continuous trace
    motor_indices = ev["Motor_location"].dropna().index.values
    motor_events = ev.iloc[motor_indices]["Motor_location"].values
    motor_event_times = ev.iloc[motor_indices]["time"].values

    motor_values_array = np.zeros_like(master_times_array)
    evt_times = np.asarray(motor_event_times, dtype=float)
    evt_names = np.asarray(motor_events).astype(str)

    # Ensure events are processed in chronological order
    order = np.argsort(evt_times)
    evt_times = evt_times[order]
    evt_names = np.char.lower(evt_names[order])

    num_events = len(evt_names)
    for i in range(num_events):
        name = evt_names[i].strip()
        t0 = float(evt_times[i])
        # Use end of array for the final interval
        t1 = (
            float(evt_times[i + 1])
            if i + 1 < num_events
            else (master_times_array[-1] + 1.0 / fs)
        )

        idx0 = int(np.searchsorted(master_times_array, t0, side="left"))
        idx1 = int(np.searchsorted(master_times_array, t1, side="left"))
        if idx1 <= idx0:
            continue

        seg_times = master_times_array[idx0:idx1]
        if name == "forward":
            denom = max(t1 - t0, 1e-12)
            motor_values_array[idx0:idx1] = np.clip((seg_times - t0) / denom, 0.0, 1.0)
        elif name == "backward":
            denom = max(t1 - t0, 1e-12)
            motor_values_array[idx0:idx1] = 1.0 - np.clip(
                (seg_times - t0) / denom, 0.0, 1.0
            )
        elif name == "presenting":
            motor_values_array[idx0:idx1] = 1.0
        # unknown labels are left as-in (zeros) - i.e. anything that is not forward, backward, presenting

    # Format states
    state_indices = ev["State"].dropna().index.values
    state_events = ev.iloc[state_indices]["State"].values
    state_event_times = ev.iloc[state_indices]["time"].values
    start_s = []
    end_s = []
    labels = []
    for i, state in enumerate(state_events):
        start_s.append(state_event_times[i])
        labels.append(state)
        if i < len(state_events) - 1:
            end_s.append(state_event_times[i + 1])
        else:
            end_s.append(master_times_array[-1])
    states_df = pd.DataFrame({"start_s": start_s, "end_s": end_s, "label": labels})

    # save to the data directory
    np.save(os.path.join(data_dir, "motor_y.npy"), motor_values_array)
    np.save(os.path.join(data_dir, "motor_t.npy"), master_times_array)
    states_df.to_csv(os.path.join(data_dir, "states.csv"), index=False)
    frames = pd.read_csv(os.path.join(data_dir, frames_name))
    frame_times = frames["pi5_sync_s"].values
    np.save(os.path.join(data_dir, "frame_times.npy"), frame_times)


if st.button("Launch GUI using data in current directory"):
    traces = []
    trace_colors = []
    traces.append(os.path.join(data_dir, "motor_y.npy"))
    traces.append(os.path.join(data_dir, "motor_t.npy"))
    trace_colors.append(glu_green)

    raster_y = []
    raster_t = []
    raster_colors = []
    raster_y.append(os.path.join(data_dir, "valve_events_y.npy"))
    raster_t.append(os.path.join(data_dir, "valve_events_t.npy"))
    raster_colors.append(valve_blue)

    raster_y.append(os.path.join(data_dir, "lick_events_y.npy"))
    raster_t.append(os.path.join(data_dir, "lick_events_t.npy"))
    raster_colors.append(lick_pink)

    video_path = os.path.join(data_dir, "vid.mp4")
    frame_times_path = os.path.join(data_dir, "frame_times.npy")
    cmd = [
        sys.executable,
        app_path,
        "--data_files",
        *traces,
        "--video",
        video_path,
        "--frame_times",
        frame_times_path,
        "--matrix_timestamps",
        *raster_t,
        "--matrix_yvals",
        *raster_y,
        "--colors",
        *trace_colors,
        "--matrix_colors",
        *raster_colors,
        "--fixed_scale",
        "--low_profile_x",
    ]
    subprocess.run(cmd, check=False)
