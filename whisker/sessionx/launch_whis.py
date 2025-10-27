from trace_colors import *
import subprocess
import sys
import os
import pandas as pd
import numpy as np
import streamlit as st

app_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "sleepscoring", "sleepscore_main.py"
)


# DEFINE THESE IF FORMATTING IS NEEDED
events_name = st.text_input("Events File Name", value="SESSIONx_EVENTS.csv")
frames_name = st.text_input("Frames File Name", value="SESSIONx_FRAMES.csv")
video_name = st.text_input("Video File Name", value="SESSIONx.mp4")


if st.button("Format Data"):
    ev = pd.read_csv(events_name)
    ev.rename(columns={"Pico_gTime_s": "time"}, inplace=True)
    array_start = 0
    array_end = round(ev["time"].values.max())
    fs = 1000

    master_times_array = np.arange(array_start, array_end, 1 / fs)
    lick_times = ev.loc[ev.Lick_Detected == "Lick"]["time"].values
    valve_opens = ev.loc[ev.Valve_timing == "Open"]["time"].values
    valve_closes = ev.loc[ev.Valve_timing == "Close"]["time"].values
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
        elif name == "idle":
            motor_values_array[idx0:idx1] = 0.0
        # unknown labels are left as-in (zeros)

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

    # valves and licks
    open_indices = np.searchsorted(master_times_array, valve_opens)
    close_indices = np.searchsorted(master_times_array, valve_closes)
    valve_values = np.zeros_like(master_times_array)
    for o, c in zip(open_indices, close_indices):
        valve_values[o:c] = 1.0
    lick_times
    lick_ends = lick_times + 0.005
    lick_values = np.zeros_like(master_times_array)
    lick_indices = np.searchsorted(master_times_array, lick_times)
    lick_ends_indices = np.searchsorted(master_times_array, lick_ends)
    for lick, lick_end in zip(lick_indices, lick_ends_indices):
        lick_values[lick:lick_end] = 1.0

    # save data to this directory
    np.save("motor_y.npy", motor_values_array)
    np.save("valve_y.npy", valve_values)
    np.save("lick_y.npy", lick_values)

    np.save("motor_t.npy", master_times_array)
    np.save("valve_t.npy", master_times_array)
    np.save("lick_t.npy", master_times_array)
    states_df.to_csv("states.csv", index=False)
    frames = pd.read_csv(frames_name)
    frame_times = frames["Pico_gTime_s"].values
    np.save("frame_times.npy", frame_times)


if st.button("Launch GUI using data in current directory"):
    traces = []
    trace_colors = []
    traces.append("motor_y.npy")
    traces.append("motor_t.npy")
    trace_colors.append(glu_green)
    traces.append("valve_y.npy")
    traces.append("valve_t.npy")
    trace_colors.append(load_blue)
    traces.append("lick_y.npy")
    traces.append("lick_t.npy")
    trace_colors.append(whisk_pink)

    video_path = "SESSIONx.mp4"
    frame_times_path = "frame_times.npy"

    cmd = [
        sys.executable,
        app_path,
        "--data_files",
        *traces,
        "--video",
        video_path,
        "--frame_times",
        frame_times_path,
        "--colors",
        *trace_colors,
        "--fixed_scale",
        "--low_profile_x",
    ]
    subprocess.run(cmd, check=True)
