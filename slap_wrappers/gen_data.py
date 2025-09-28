import cv2
import numpy as np

offset = 1874.5776
vid_path = ".\\sync_block-1\\full2dmd.mp4"
# load the video and count the number of frames
cap = cv2.VideoCapture(vid_path)
num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(num_frames)
frame_times_2 = np.arange(0, num_frames) / 20 + offset
print(frame_times_2.min(), frame_times_2.max())
np.save("frame_times_2.npy", frame_times_2)
