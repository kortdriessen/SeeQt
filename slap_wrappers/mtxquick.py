import glob
import os
import subprocess
import sys

app_path = "C:\\Users\\driessen2\\python_projects\\DATA_VIEWERS\\SeeQt\\sleepscoring\\sleepscore_main.py"

traces = []
eeg_t_path = (
    "C:\\Users\\driessen2\\python_projects\\DATA_VIEWERS\\SeeQt\\data\\eeg_t.npy"
)
eeg_y_path = (
    "C:\\Users\\driessen2\\python_projects\\DATA_VIEWERS\\SeeQt\\data\\eeg_y.npy"
)
traces.append(eeg_y_path)
traces.append(eeg_t_path)

load_t_path = (
    "C:\\Users\\driessen2\\python_projects\\DATA_VIEWERS\\SeeQt\\data\\load_t.npy"
)
load_y_path = (
    "C:\\Users\\driessen2\\python_projects\\DATA_VIEWERS\\SeeQt\\data\\load_y.npy"
)
traces.append(load_y_path)
traces.append(load_t_path)


video_path = (
    "C:\\Users\\driessen2\\python_projects\\DATA_VIEWERS\\SeeQt\\data\\video.mp4"
)
frame_times_path = (
    "C:\\Users\\driessen2\\python_projects\\DATA_VIEWERS\\SeeQt\\data\\frame_times.npy"
)


matrix_times = [".\\test_mat\\ev_times_0.npy", ".\\test_mat\\ev_times_1.npy"]
matrix_yvals = [".\\test_mat\\ev_sources_0.npy", ".\\test_mat\\ev_sources_1.npy"]
matrix_alphas = [".\\test_mat\\ev_alphas_0.npy", ".\\test_mat\\ev_alphas_1.npy"]
matrix_colors = ["#4287f5", "#f54254"]

mtx_heights = ["0.0005", "0.0005"]

low_profile_x_arg = "--low_profile_x"
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
    "white",
    "white",
    "--fixed_scale",
    low_profile_x_arg,
    "--matrix_timestamps",
    *matrix_times,
    "--matrix_yvals",
    *matrix_yvals,
    "--alpha_vals",
    *matrix_alphas,
    "--matrix_colors",
    *matrix_colors,
    "--matrix_row_height",
    "0.0001",
]

subprocess.run(cmd, check=True)
