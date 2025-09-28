import glob
import os

slap_directory_root = "Z:\\slap_mi"
subject = "kaus"
exp = "exp_1"
sync_block = 1


scoring_data_dir = f"{slap_directory_root}\\analysis_materials\\{subject}\\{exp}\\scoring_data\\sync_block-{sync_block}"
frame_times_paths = glob.glob(f"{scoring_data_dir}\\*frame_times.npy")
assert (
    len(frame_times_paths) == 1
), "Expected 1 frame times path, got {len(frame_times_paths)}"
frame_times_path = frame_times_paths[0]
pupil_path = (
    f"{slap_directory_root}\\data\\{subject}\\{exp}\\pupil\\pupil-{sync_block}.mp4"
)

os.system(
    f"python sleepscore_main.py --data_dir {scoring_data_dir} --video {pupil_path} --frame_times {frame_times_path}"
)
