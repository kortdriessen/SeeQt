import os
import sys

path_to_data = r".\\data\\"
path_to_video = r".\\data\\video.mp4"
path_to_frame_times = r".\\data\\frame_times.npy"
path_to_image = r".\\data\\stat_im.png"
path_to_video2 = r".\\data\\video2.mp4"
path_to_frame_times2 = r".\\data\\frame_times2.npy"

os.system(
    f"python sleepscore_main.py --data_dir {path_to_data} --video {path_to_video} --frame_times {path_to_frame_times} --video2 {path_to_video2} --frame_times2 {path_to_frame_times2}"
)
