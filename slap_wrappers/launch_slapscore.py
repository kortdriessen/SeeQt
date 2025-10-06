import yaml
import os
import glob
import subprocess
import sys
from trace_colors import *
import streamlit as st
from slap_win_utils import *
from PIL import Image

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

rois = st.checkbox("Include Soma ROIs", value=False)
if rois:
    all_rois = st.checkbox("include ROIs for all acquisitions?", value=False)
synapses = st.checkbox("Include Synaptic Traces", value=False)


traces = []
trace_colors = []

if acq_id is not None:
    loc, acq = acq_id.split("--")
    for dmd in [1, 2]:
        mean_im_path = f"Z:\\slap_mi\\analysis_materials\\{subject}\\{exp}\\mean_IMs\\{loc}\\{acq}\\DMD-{dmd}__labelled.png"
        if not os.path.exists(mean_im_path):
            mean_im_path = f"Z:\\slap_mi\\analysis_materials\\{subject}\\{exp}\\mean_IMs\\{loc}\\{acq}\\DMD-{dmd}.png"
        if not os.path.exists(mean_im_path):
            st.write(f"Mean image not found: {mean_im_path}")
            continue
        im = Image.open(mean_im_path)
        st.image(im, use_container_width=False)


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
    if all_rois:
        for ac in acq_ids:
            roi_data_dir = f"Z:\\slap_mi\\analysis_materials\\{subject}\\{exp}\\scoring_data\\sync_block-{sb}\\scope_traces\\soma_rois\\{ac}"
            for f in os.listdir(roi_data_dir):
                if not f.endswith("_y.npy"):
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
    else:
        roi_data_dir = f"Z:\\slap_mi\\analysis_materials\\{subject}\\{exp}\\scoring_data\\sync_block-{sb}\\scope_traces\\soma_rois\\{acq_id}"
        for f in os.listdir(roi_data_dir):
            if not f.endswith("_y.npy"):
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
pupil_path = f"Z:\\slap_mi\\data\\{subject}\\{exp}\\pupil\\pupil-{sb}.mp4"


# get the synaptic data
if synapses:
    for dmd in [1, 2]:
        syn_data_dir = f"Z:\\slap_mi\\analysis_materials\\{subject}\\{exp}\\scoring_data\\sync_block-{sb}\\scope_traces\\synapses\\{acq_id}\\dmd{dmd}"
        syn_files = glob.glob(os.path.join(syn_data_dir, "*_y.npy"))
        for syn_file in syn_files:
            traces.append(os.path.join(syn_data_dir, syn_file))
            time_file = syn_file.replace("_y.npy", "_t.npy")
            traces.append(os.path.join(syn_data_dir, time_file))
            trace_colors.append(glu_green)

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
    subprocess.run(cmd, check=True)
