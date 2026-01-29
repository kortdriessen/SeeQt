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
import numpy as np

app_path = "C:\\Users\\driessen2\\python_projects\\DATA_VIEWERS\\SeeQt\\sleepscoring\\sleepscore_main.py"

directory = (
    "L:\\Data\\ACR_PROJECT_MATERIALS\\plots_presentations_etc\\data_browse\\ACR_40--swi"
)

traces = []
mtx_dat = []
mtx_t = []
mtx_a = []

for f in os.listdir(directory):
    print(f)
    if "ts_y.npy" in f:
        traces.append(os.path.join(directory, f))
        traces.append(os.path.join(directory, f.replace("ts_y.npy", "ts_t.npy")))
    elif "mtx_y.npy" in f:
        mtx_dat.append(os.path.join(directory, f))
        mtx_t.append(os.path.join(directory, f.replace("mtx_y.npy", "mtx_t.npy")))
        mtx_a.append(os.path.join(directory, f.replace("mtx_y.npy", "mtx_a.npy")))

matrix_colors = [
    "#0362fc",
    "#f5f7fa",
]
trace_colors = [
    "#0362fc",
    "#f5f7fa",
]

print(traces)

cmd = [
    sys.executable,
    app_path,
    "--data_files",
    *traces,
    "--colors",
    *trace_colors,
    "--fixed_scale",
    "--low_profile_x",
    "--matrix_timestamps",
    *mtx_t,
    "--matrix_yvals",
    *mtx_dat,
    "--alpha_vals",
    *mtx_a,
    "--matrix_colors",
    *matrix_colors,
]
subprocess.run(cmd, check=True)
