import yaml
import os
import glob
import subprocess
import sys
from trace_colors import *
import streamlit as st
from slap_win_utils import *
from PIL import Image
import pandas as pd

app_path = "C:\\Users\\driessen2\\python_projects\\DATA_VIEWERS\\SeeQt\\sleepscoring\\sleepscore_main.py"


@st.cache_data
def load_ei():
    return load_exp_info()


@st.cache_data
def load_si():
    return load_sync_info()


exp_info = load_ei()
si = load_si()


subjects = sorted(exp_info["subject"].unique().to_list())
subject = st.selectbox("Subject", subjects, key="subject")

subject_exps = sorted(
    exp_info.filter(pl.col("subject") == st.session_state.subject)["experiment"]
    .unique()
    .to_list()
)
exp = st.selectbox("Experiment", subject_exps, key="experiment")

sync_blocks = list(si[subject][exp]["sync_blocks"].keys())

sb = st.selectbox("Sync Block", sync_blocks, key="sync_block")

acq_ids = [
    acq
    for acq in si[subject][exp]["acquisitions"]
    if si[subject][exp]["acquisitions"][acq]["sync_block"] == sb
]

acq_id = st.selectbox("Acquisition ID for ROIs", acq_ids, key="acq_id")
loc, acq = acq_id.split("--")
rois = st.checkbox("Include Soma ROIs", value=False)

eye = st.checkbox("Include Eye Traces", value=False)
whisking = st.checkbox("Include Whisking Traces", value=False)
glut_sums = st.checkbox("Include Glutamate Sums", value=False)
fractional = st.checkbox("Fraction Active?", value=False)
soma = st.text_input("Soma to Use", value="soma1")

synaptic_videos = st.checkbox("Include Synaptic Videos", value=False)

if synaptic_videos:
    video_2_path = f"Z:\\slap_mi\\analysis_materials\\plots\\movies\\{subject}\\{exp}\\{loc}\\{acq}\\full\\imaging_data_DMD1.mp4"
    video_3_path = f"Z:\\slap_mi\\analysis_materials\\plots\\movies\\{subject}\\{exp}\\{loc}\\{acq}\\full\\imaging_data_DMD2.mp4"

    frame_times_2_path = f"Z:\\slap_mi\\analysis_materials\\plots\\movies\\{subject}\\{exp}\\{loc}\\{acq}\\full\\imaging_frame_times.npy"
    frame_times_3_path = f"Z:\\slap_mi\\analysis_materials\\plots\\movies\\{subject}\\{exp}\\{loc}\\{acq}\\full\\imaging_frame_times.npy"

traces = []
trace_colors = []

# if acq_id is not None:
#    loc, acq = acq_id.split("--")
#    for dmd in [1, 2]:
#        mean_im_path = f"Z:\\slap_mi\\analysis_materials\\{subject}\\{exp}\\mean_IMs\\{loc}\\{acq}\\DMD-{dmd}__labelled.png"
#        if not os.path.exists(mean_im_path):
#            mean_im_path = f"Z:\\slap_mi\\analysis_materials\\{subject}\\{exp}\\mean_IMs\\{loc}\\{acq}\\DMD-{dmd}.png"
#        if not os.path.exists(mean_im_path):
#            st.write(f"Mean image not found: {mean_im_path}")
#            continue
#        im = Image.open(mean_im_path)
#        st.image(im, use_container_width=False)


main_data_dir = f"Z:\\slap_mi\\analysis_materials\\{subject}\\{exp}\\scoring_data\\sync_block-{sb}\\"
for f in os.listdir(main_data_dir):
    if f.endswith("_y.npy"):
        if "EEG" in f:
            traces.append(os.path.join(main_data_dir, f))
            time_file = f.replace("_y.npy", "_t.npy")
            traces.append(os.path.join(main_data_dir, time_file))
            trace_colors.append(trace_white)

for f in os.listdir(main_data_dir):
    if f.endswith("_y.npy"):
        if "loal" in f:
            traces.append(os.path.join(main_data_dir, f))
            time_file = f.replace("_y.npy", "_t.npy")
            traces.append(os.path.join(main_data_dir, time_file))
            trace_colors.append(load_blue)

if rois:
    roi_data_dir = f"Z:\\slap_mi\\analysis_materials\\{subject}\\{exp}\\scoring_data\\sync_block-{sb}\\scope_traces\\soma_rois\\{acq_id}"
    for f in os.listdir(roi_data_dir):
        if not f.endswith("_y.npy"):
            continue
        if soma not in f:
            continue
        if "dmd1" in f:
            trace_colors.append(soma_red1)
        elif "dmd2" in f:
            trace_colors.append(soma_red2)
        else:
            trace_colors.append(soma_red1)

        yname = f
        tname = f.replace("_y.npy", "_t.npy")
        roi_data_path = os.path.join(roi_data_dir, yname)
        roi_time_path = os.path.join(roi_data_dir, tname)
        traces.append(roi_data_path)
        traces.append(roi_time_path)

if glut_sums:

    glut_sums_dir = f"Z:\\slap_mi\\analysis_materials\\{subject}\\{exp}\\scoring_data\\sync_block-{sb}\\scope_traces\\synapses\\{acq_id}\\glutamate_sums"
    glut_sums_data = [
        f for f in os.listdir(glut_sums_dir) if f.endswith("_y.npy") and soma in f
    ]
    for f in glut_sums_data:
        yname = f
        tname = f.replace("_y.npy", "_t.npy")
        traces.append(os.path.join(glut_sums_dir, yname))
        traces.append(os.path.join(glut_sums_dir, tname))
        trace_colors.append(glut_total_green)

    glut_sums_dir = f"Z:\\slap_mi\\analysis_materials\\{subject}\\{exp}\\scoring_data\\sync_block-{sb}\\scope_traces\\synapses\\{acq_id}\\glutamate_sums\\BY_DEND\\{soma}"
    glut_sums_data = [
        f for f in os.listdir(glut_sums_dir) if f.endswith("_y.npy") and "fr_" not in f
    ]
    glut_sums_frac = [
        f for f in os.listdir(glut_sums_dir) if f.endswith("_y.npy") and "fr_" in f
    ]
    for i, f in enumerate(glut_sums_data):
        yname = f
        tname = f.replace("_y.npy", "_t.npy")
        if fractional:
            yname = glut_sums_frac[i]
            tname = yname.replace("_y.npy", "_t.npy")
        traces.append(os.path.join(glut_sums_dir, yname))
        traces.append(os.path.join(glut_sums_dir, tname))
        trace_colors.append(glut_sum_green)


# Validate inputs early
for i in range(len(traces)):
    assert os.path.exists(traces[i]), f"File does not exist: {traces[i]}"

# get frames times and pupil video
frame_times_paths = glob.glob(
    f"Z:\\slap_mi\\analysis_materials\\{subject}\\{exp}\\scoring_data\\sync_block-{sb}\\*frame_times.npy"
)
assert (
    len(frame_times_paths) == 1
), "Expected 1 frame times path, got {len(frame_times_paths)}"

frame_times_path = frame_times_paths[0]
# pupil_path = f"Z:\\slap_mi\\data\\{subject}\\{exp}\\pupil\\pupil-{sb}.mp4"
pupil_path = f"Z:\\slap_mi\\analysis_materials\\{subject}\\{exp}\\pupil_inference\\pupil-{sb}DLC_Resnet50_dlc_slap_pupilSep23shuffle0_snapshot_best-60_p60_labeled.mp4"

if eye:
    eye_data_dir = f"Z:\\slap_mi\\analysis_materials\\{subject}\\{exp}\\scoring_data\\sync_block-{sb}\\eye"
    for f in os.listdir(eye_data_dir):
        if not f.endswith("_y.npy"):
            continue
        if "lid" in f:
            continue  # don't display lid data

        traces.append(os.path.join(eye_data_dir, f))
        time_file = f.replace("_y.npy", "_t.npy")
        traces.append(os.path.join(eye_data_dir, time_file))
        trace_colors.append(eye_indigo)

if whisking:
    whisk_dir = f"Z:\\slap_mi\\analysis_materials\\{subject}\\{exp}\\scoring_data\\sync_block-{sb}\\whisking"
    for f in os.listdir(whisk_dir):
        if not f.endswith("_y.npy"):
            continue
        traces.append(os.path.join(whisk_dir, f))
        time_file = f.replace("_y.npy", "_t.npy")
        traces.append(os.path.join(whisk_dir, time_file))
        trace_colors.append(whisk_pink)

write_temp_hypno = st.sidebar.checkbox("Write temp hypnogram to file", value=False)
if write_temp_hypno:
    temp_hypno_path = f"Z:\\slap_mi\\analysis_materials\\{subject}\\{exp}\\scoring_data\\sync_block-{sb}\\hypnograms\\model_labelled\\raw_epochs.csv"
    if not os.path.exists(temp_hypno_path):
        st.write(f"Temp hypnogram file not found: {temp_hypno_path}")
        write_temp_hypno = False

    write_path = "temp_hypnogram.csv"

    keep_state_threshold = st.sidebar.number_input(
        "Keep state threshold (must be above this to maintain model prediction)",
        value=0.5,
        min_value=0.01,
        max_value=0.99,
    )
    keep_state_threshold = float(keep_state_threshold)

    if st.sidebar.button("Apply threshold and write hypnogram"):
        if os.path.exists(write_path):
            st.write(f"Removing existing temp hypnogram: {write_path}")
            os.remove(write_path)

        epoch_df = pd.read_csv(temp_hypno_path)
        epoch_df["label"] = epoch_df["label"].mask(
            epoch_df[["NREM_smooth", "REM_smooth", "Wake_smooth"]].max(axis=1)
            < keep_state_threshold,
            "unclear",
        )
        # merge adjacent epochs with the same label
        epoch_df_sorted = epoch_df.sort_values("start_s").reset_index(drop=True)
        # Create a group identifier for consecutive labels
        epoch_df_sorted["group"] = (
            epoch_df_sorted["label"] != epoch_df_sorted["label"].shift()
        ).cumsum()
        # Group by the label and group identifier, then aggregate
        merged_epoch_df = (
            epoch_df_sorted.groupby(["label", "group"])
            .agg({"start_s": "first", "end_s": "last"})
            .reset_index()
        )
        # Drop the temporary group column
        merged_epoch_df = merged_epoch_df.drop("group", axis=1)
        # Sort by start_s for final output
        merged_epoch_df = merged_epoch_df.sort_values("start_s").reset_index(drop=True)
        final_hypno = merged_epoch_df[["start_s", "end_s", "label"]]
        final_hypno.to_csv(write_path, index=False)
        st.write(f"Temp hypnogram written to {write_path}")

low_profile_x = st.checkbox("Low Profile X", value=False)


if st.button("Launch GUI"):
    # Build argument list and invoke without shell quoting issues
    if low_profile_x:
        low_profile_x_arg = "--low_profile_x"
    else:
        low_profile_x_arg = ""
    cmd = [
        sys.executable,
        app_path,
        "--data_files",
        *traces,
        "--video",
        pupil_path,
        "--frame_times",
        frame_times_path,
        "--colors",
        *trace_colors,
        "--fixed_scale",
        low_profile_x_arg,
    ]
    if synaptic_videos:
        cmd.extend(
            [
                "--video2",
                video_2_path,
                "--frame_times2",
                frame_times_2_path,
                "--video3",
                video_3_path,
                "--frame_times3",
                frame_times_3_path,
            ]
        )
    subprocess.run(cmd, check=True)
