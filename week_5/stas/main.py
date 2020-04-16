# %%
from itertools import islice
import os
from pathlib import Path
import random
import sys

import cv2
import imageio
import numpy as np
from tqdm.auto import tqdm

from bbox import BBox
from colors import colors_panetone
from reader import TrackReader, Video_Detection
from utils import draw_detections, visualize_multi_cam
from video import Video, VideoSet

# CONFIG. Change to your needs!

# Path to the root of AIC20_track3 dataset
base_dir = Path("/data/AIC20_track3")
# Path for storing the results
results_dir = Path("/home/malpunek/shit/m6results")


# END CONFIG

os.makedirs(results_dir, exist_ok=True)

# %%
# Some function to get path to objects


def cam_dir(sequence, cam, train=True):
    return (
        base_dir / ("train" if train else "test") / f"S{sequence:02d}" / f"c{cam:03d}"
    )


def our_detection(seq, cam, method="mask_rcnn_ft"):
    available_methods = ["mask", "mask_rcnn_ft", "retinanet", "yolo"]
    assert (
        method in available_methods
    ), f"Please make sure method is one of {available_methods}"
    return Path(
        f"../detections_dl/s{seq:02d}/{method}/detections_{method}_c{cam:03d}_S{seq:02d}.txt"
    )


def our_tracking(seq, cam, method="mask_rcnn_ft"):
    available_methods = ["mask", "mask_rcnn_ft", "retinanet", "yolo"]
    assert (
        method in available_methods
    ), f"Please make sure method is one of {available_methods}"
    return Path(
        f"../mtsc/detections_nn/S{seq:02d}/detections_{method}_c{cam:03d}_Overlap_S{seq:02d}.pkl"
    )


# %%
# (Un)distortion
seq, cam = 4, 35  # One of the few distorted cameras
work_dir = cam_dir(seq, cam)

vid = Video(work_dir / "vdo.avi")
bbox = BBox(200, 200, 500, 500)

dist = next(iter(vid.frames))
l, t, r, b = bbox.ltrb
cv2.rectangle(dist, (l, t), (r, b), [255, 0, 0])

und = next(iter(vid))
l, t, r, b = bbox.undistorted(vid.camera_matrix, vid.distortion_coeffs).ltrb.astype(int)
cv2.rectangle(und, (l, t), (r, b), [255, 0, 0])

cv2.imshow("DISTORTED", dist)
cv2.imshow("UNDISTORTED", und)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
# Init Single Camera

seq, cam = 1, 1
work_dir = cam_dir(seq, cam)

# Pick the detection and tracking
# detection = Video_Detection(work_dir / "det" / "det_yolo3.txt")
detection = Video_Detection(our_detection(seq, cam))

# NOTE: the class named Video_Detection works also for tracking
# but, our tracks are stored as .pkl files

# tracking = Video_Detection(work_dir / "mtsc" / "mtsc_tc_mask_rcnn.txt")
tracking = TrackReader(our_tracking(seq, cam))

video_path = work_dir / "vdo.avi"

vid = Video(video_path, detection_method=detection, tracking_method=tracking)

# %%
# Detection
vid.eval_detection()
vid.dump_visualization(
    results_dir / "our_detections.mp4",
    detection,
    fps=1,
    begin=50,
    end=200,
    with_gt=True,
)


# %%
# Tracking

vid.eval_tracking(tracking, iou_threshold=0.7)
vid.dump_visualization(
    results_dir / "our_tracking.mp4", tracking, fps=5, begin=50, end=200, with_gt=True
)

# %%
# Single-Camera Tracking With Map Visualization

work_dir = cam_dir(1, 1)

video_path = work_dir / "vdo.avi"
detection = Video_Detection(work_dir / "gt" / "gt.txt")
tracking = Video_Detection(work_dir / "mtsc" / "mtsc_deepsort_yolo3.txt")

vid = Video(
    video_path, detection_method=detection, tracking_method=tracking, verbose=True
)

vid.dump_visualized_map(
    results_dir / "tracking_map.mp4",
    50,
    300,
    fps=10,
    lasting=True,
    map_frame_shape=(400, 600),
)


# %%
# Multi-Camera init

# Pick a sequence
sequence = 1

all_cameras_sequence = {
    1: list(range(1, 6)),
    3: list(range(10, 16)),
    4: list(range(16, 41)),
}

cam_list = all_cameras_sequence[sequence]
work_dirs = [cam_dir(sequence, i) for i in cam_list]
video_paths = [wd / "vdo.avi" for wd in work_dirs]
gt_paths = [wd / "gt" / "gt.txt" for wd in work_dirs]
track_paths = [our_tracking(sequence, cam) for cam in cam_list]

videos = [
    Video(vp, detection_method=Video_Detection(gtp), tracking_method=TrackReader(trp))
    for vp, gtp, trp in zip(video_paths, gt_paths, track_paths)
]

vs = VideoSet(videos)


# %%
# Multi-Camera tracking visualization without map

it_frame = tqdm(islice(iter(vs), 200, 250), file=sys.stdout, total=50)
it_track = islice(vs.tracking(), 200, 250)

writer = imageio.get_writer(results_dir / "tracking_multi.mp4", fps=2)
clrs = np.array(colors_panetone).tolist()
random.shuffle(clrs)

for frame, tracks in zip(it_frame, it_track):
    frame = draw_detections(frame, tracks, colors=clrs)
    writer.append_data(frame)

writer.close()


# %%
# Multi-Camera evaluation

vs.eval_tracking(use_encoding=False)


# %%
# Test Multi Cam Visualization


# If there's nothing on the minimap:
# For cameras 1-5 use:
#   - pix_per_10_meters=50
#   - shift=(0, 0)
# For others: chech the "results / positions.txt" file. And adjust
# these to so they fit within (0,0) x (400,400)
# pix_per_10_meters: 0.0001 degree is about 10 meters in reality.
#     This param sets how big is the differenve of 10 meters on minimap
# shift: cameras 16-40 are mostly located far from the center of the scenario
#     Look at the file save_map_positions. It's a mapping from obj_id into
#     mini-map positions.

visualize_multi_cam(
    results_dir / "multi_cam_vis.mp4",
    vs,
    begin=0,
    end=300,
    save_map_positions=results_dir / "positions.txt",
    pix_per_10_meters=5,
    map_shift=(0, 0),
)


# %%
