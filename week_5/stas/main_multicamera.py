# flake8: noqa
# %%
from itertools import islice
import os
from pathlib import Path
import numpy as np
from reader import Video_Detection
from video import Video, VideoSet
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image

from kalman_filter_gps import TrackList
from reader import TrackReader


# CONFIG. Change to your needs!
visualize = False
use_encoding = False
base_dir = Path("/data/AIC20_track3")
results_dir = Path("/home/malpunek/shit/m6results")
# END CONFIG


def gps2pixel_map(lat, lon):
    """
    42.526954, -90.726756 top left
    42.523530, -90.720147 bottom right
    image: 1233x870
    """
    top = 42.526954
    bottom = 42.523530
    left = -90.726756
    right = -90.720147
    width = 1233
    height = 870
    x_normalized = (lon - left) / (right - left)
    y_normalized = (lat - top) / (bottom - top)
    return x_normalized * width, y_normalized * height


os.makedirs(results_dir, exist_ok=True)


def cam_dir(sequence, cam, train=True):
    return (
        base_dir / ("train" if train else "test") / f"S{sequence:02d}" / f"c{cam:03d}"
    )


def our_tracking(seq, cam, method="mask_rcnn_ft"):
    available_methods = ["mask", "mask_rcnn_ft", "retinanet", "yolo"]
    assert (
        method in available_methods
    ), f"Please make sure method is one of {available_methods}"
    return Path(
        f"../mtsc/detections_nn/S{seq:02d}/detections_{method}_c{cam:03d}_Overlap_S{seq:02d}.pkl"
    )


def plot_detections_camera(detections, ax, cmap, image, timestamp):
    ax.clear()
    for detection in detections:
        id = detection.obj_id
        caption_id = "ID: {}".format(id)
        color = cmap(id % num_color)
        left, top, width, height = detection.ltwh
        bbox = patches.Rectangle(
            (left, top),
            width,
            height,
            linewidth=2,
            edgecolor=color,
            fill=False,
            facecolor=color,
        )
        ax.add_patch(bbox)
        ax.text(left, top - 16, caption_id, color=color)
    ax.text(10, -20, f"Time: {timestamp:.2f}", color="black", fontsize=11)
    ax.axis("off")
    ax.imshow(image)
    plt.show()
    # plt.savefig("mtmc_images/image_"+str(frame_num).zfill(3))
    plt.pause(0.01)


def plot_tracks_map(tracks, ax_map, cmap, im_map, camera_num=1):
    ax_map.clear()
    for track in tracks:
        lat, lon = track[0]
        id = track[1]
        caption_id = "ID: {}".format(id)
        x_map, y_map = gps2pixel_map(lat=lat, lon=lon)
        color = cmap(id % num_color)
        square_size = 10
        bbox = patches.Rectangle(
            (x_map - square_size / 2, y_map - square_size / 2),
            square_size,
            square_size,
            linewidth=1,
            edgecolor=color,
            fill=True,
            facecolor=color,
        )
        ax_map.add_patch(bbox)
        ax_map.text(x_map, y_map - 16, caption_id, color=color)
    ax_map.text(
        10,
        -20,
        f"Time: {timestamp:.3f},   Camera: {camera_num}",
        color="black",
        fontsize=11,
    )
    ax_map.axis("off")
    ax_map.imshow(im_map)
    plt.show()
    # plt.savefig("mtmc_images/image_"+str(frame_num).zfill(3))
    plt.pause(0.01)


# %%
# Test Multi Cam Visualization
# Sequence 1
all_cameras_sequence = {
    1: list(range(1, 6)),
    3: list(range(10, 16)),
    4: list(range(16, 41)),
}
sequences = [1, 3, 4]
for sequence in sequences:
    cam_list = all_cameras_sequence[sequence]
    work_dirs = [cam_dir(sequence, i) for i in cam_list]
    video_paths = [wd / "vdo.avi" for wd in work_dirs]
    gt_paths = [wd / "gt" / "gt.txt" for wd in work_dirs]
    pkl_paths = [our_tracking(sequence, cam) for cam in cam_list]
    videos = [
        Video(
            vp,
            detection_method=Video_Detection(gtp),
            tracking_method=TrackReader(pklp)
        )
        for vp, gtp, pklp in zip(video_paths, gt_paths, pkl_paths)
    ]

    vs = VideoSet(videos)

    cmap = plt.cm.get_cmap("Dark2")
    num_color = cmap.N
    ax_map = plt.subplot(232)
    ax_map_tracks = plt.subplot(235)
    ax_cameras = {
        1: plt.subplot(231),
        2: plt.subplot(233),
        3: plt.subplot(234),
        4: plt.subplot(236),
    }

    plt.ion()

    if visualize:
        path_map = str(work_dirs[0]) + "/../reference_image.png"
        track_list = TrackList(encoding=use_encoding)
        im_map = np.array(Image.open(path_map), dtype=np.uint8)
        for gps, vid, timestamp, detections, image in zip(
            vs.gps_positions(with_ids=True),
            vs.video_objs(),
            vs.timestamps(),
            vs.tracking(),
            vs,
        ):
            if use_encoding:
                camera_tracks = []
                for gps_track, bbox in zip(gps, detections):
                    camera_tracks.append([gps_track[0], gps_track[1], bbox])
                id_map, gps_global_ids = track_list.process_detections(
                    detections=camera_tracks, time=timestamp, image=image
                )
            else:
                camera_tracks = list(gps)
                id_map, gps_global_ids = track_list.process_detections(
                    detections=camera_tracks, time=timestamp
                )
            if visualize:
                plot_detections_camera(
                    detections=detections,
                    ax=ax_cameras[vid.camera_num],
                    cmap=cmap,
                    image=image,
                    timestamp=timestamp,
                )
                plot_tracks_map(
                    tracks=gps_global_ids,
                    ax_map=ax_map,
                    cmap=cmap,
                    im_map=im_map,
                    camera_num=vid.camera_num,
                )
                plot_tracks_map(
                    tracks=track_list.get_list(),
                    ax_map=ax_map_tracks,
                    cmap=cmap,
                    im_map=im_map,
                    camera_num=vid.camera_num,
                )
    else:
        print(f"Evaluating SEQ using our detection: {sequence}")
        IDF1_score = vs.eval_tracking(use_encoding=use_encoding)
