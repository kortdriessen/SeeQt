from trace_colors import *
import subprocess
import sys
import os

app_path = "C:\\Users\\driessen2\\python_projects\\DATA_VIEWERS\\SeeQt\\sleepscoring\\sleepscore_main.py"
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
