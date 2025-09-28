import glob
import os
import cv2

slap_directory_root = "Z:\\slap_mi"
subject = "kaus"
exp = "exp_1"
sync_block = 1

vp2 = "..\\sync_block-1\\full2dmd.mp4"
frame_times_2_path = "..\\sync_block-1\\frame_times_2.npy"

scoring_data_dir = r"C:\\Users\\driessen2\\python_projects\\DATA_VIEWERS\\sync_block-1"
frame_times_paths = glob.glob(f"{scoring_data_dir}\\*frame_times.npy")
assert (
    len(frame_times_paths) == 1
), "Expected 1 frame times path, got {len(frame_times_paths)}"
frame_times_path = frame_times_paths[0]
pupil_path = (
    r"C:\\Users\\driessen2\\python_projects\\DATA_VIEWERS\\sync_block-1\\pupil-1.mp4"
)

os.system(
    f"python sleepscore_main.py --data_dir {scoring_data_dir} --video {pupil_path} --frame_times {frame_times_path} --video2 {vp2} --frame_times2 {frame_times_2_path}"
)
