from itertools import islice
from numbers import Number
from pathlib import Path
import random
import sys

import cv2
import imageio
import numpy as np
from tqdm.auto import tqdm

from colors import colors_panetone


def get_timestamp(video_path):
    video_path = Path(video_path)
    cam_num = int(video_path.parent.name[1:])
    seq_num = int(video_path.parent.parent.name[1:])
    timestamp_file = (
        video_path.parent.parent.parent.parent / "cam_timestamp" / f"S{seq_num:02d}.txt"
    )
    with open(timestamp_file, "rt") as ts:
        for line in ts:
            if line.startswith(f"c{cam_num:03d}"):
                return float(line.split()[-1])


def undistort_points(pts, camera_matrix, distortion_coeffs):
    pts = pts.copy().astype(float)
    pts = cv2.undistortPoints(pts, camera_matrix, distortion_coeffs)
    pts = pts.reshape((-1, 2))
    pts = np.hstack((pts, np.ones((pts.shape[0], 1))))
    pts = (camera_matrix @ pts.T).T
    pts = pts[:, 0:2] / pts[:, 2, np.newaxis]
    return pts.astype(int)


def pixel2coords(p, H):
    # TODO enforce np.longdouble for extra precision
    p = np.array([p[0], p[1], 1])
    gps = H @ p
    gps /= gps[2]
    return tuple(gps[:2])


def match_bboxes(left, right, iou_threshold):
    left, right = set(left), set(right)
    while left and right:
        best_pairs = [(l, l.sort_matches(right)[0]) for l in left]
        best_l, best_r = sorted(best_pairs, key=lambda x: x[0] @ x[1])[0]
        if best_l @ best_r < iou_threshold:
            return
        yield best_l, best_r
        left.remove(best_l)
        right.remove(best_r)


def match_detections(detections, ground_truth, iou_threshold):
    detections = sorted(detections, key=lambda b: b.confidence)
    ground_truth = set(ground_truth)
    for d in detections:
        if len(ground_truth) == 0:
            return
        match = d.sort_matches(ground_truth)[0]
        if d @ match > iou_threshold:
            yield (d, match)
            ground_truth.remove(match)
        else:
            yield (d, None)


def draw_detections(
    frame, bboxes, text=None, text_color=None, colors=colors_panetone, draw_ids=True
):
    for bbox, color in zip(bboxes, colors):
        # Apply same color to same object ids
        if isinstance(bbox.obj_id, Number) and bbox.obj_id != -1:
            color = colors[bbox.obj_id % len(colors)]
        l, t, r, b = bbox.ltrb.astype(int)
        thickness = int(bbox.confidence * 6)
        cv2.rectangle(frame, (l, t), (r, b), color, thickness=thickness)
        if text is not None or draw_ids:
            color = text_color or color
            bbox_text = text or ""
            if draw_ids:
                bbox_text = f"{bbox_text} ID: {bbox.obj_id}"
            cv2.putText(
                frame,
                bbox_text,
                (l, t),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                color,
                thickness=2,
            )
    return frame


def visualize_multi_cam(
    save_path,
    video_set,
    begin=None,
    end=None,
    target_shape=(1080, 1920),
    map_frame_shape=(400, 400),
    map_shift=(0, 0),
    pix_per_10_meters=50,
    fps=1,
    save_map_positions=None,
    verbose=True,
):

    # Init
    it_frame = iter(video_set)
    it_box = video_set.tracking()
    it_pos = video_set.map_positions(
        map_frame_shape=map_frame_shape, pix_per_10_meters=pix_per_10_meters
    )

    if begin or end:
        begin = begin or 0
        assert end is not None, "You have to supply :arg:end if you supply :arg:begin"
        it_frame = islice(it_frame, begin, end)
        if verbose:
            it_frame = tqdm(it_frame, file=sys.stdout, total=end - begin)
        it_box = islice(it_box, begin, end)
        it_pos = islice(it_pos, begin, end)
    elif verbose:
        it_frame = tqdm(it_frame, file=sys.stdout)

    writer = imageio.get_writer(save_path, fps=fps)

    clrs = np.array(colors_panetone).tolist()
    random.shuffle(clrs)
    map_frame = np.ones(map_frame_shape + (3,)) * 255

    if save_map_positions is not None:
        map_file = open(save_map_positions, "wt")

    # Actual work
    for frame, map_positions, bboxes in zip(it_frame, it_pos, it_box):
        bboxes = list(bboxes)
        frame = draw_detections(frame, bboxes, colors=clrs)
        for (map_position, obj_id), box in zip(map_positions, bboxes):

            # Pick consistent colors for objs with valid ids
            if isinstance(obj_id, Number) and obj_id != -1:
                color = clrs[obj_id % len(clrs)]
            else:
                color = random.choice(clrs)

            # Draw the same position on the picture and on the mini-map
            gx, gy = box.ground_pos()
            cv2.drawMarker(frame, (int(gx), int(gy)), color, thickness=4)
            mx, my = map_position
            mx, my = mx + map_shift[0], my + map_shift[1]
            
            if save_map_positions is not None:
                map_file.write(f"{obj_id} -> {mx, my}\n")

            cv2.drawMarker(map_frame, (mx, my), color, thickness=4)

        # The videos are in different resolutions, but we need frames of the same size
        lack_vert, lack_horiz = (
            max(target_shape[0] - frame.shape[0], 0),
            max(target_shape[1] - frame.shape[1], 0),
        )
        if lack_vert or lack_horiz:
            frame = cv2.copyMakeBorder(
                frame, lack_vert, lack_vert, lack_horiz, lack_horiz, cv2.BORDER_CONSTANT
            )
        if frame.shape[:2] != target_shape:
            y, x = frame.shape[:2]
            startx = x // 2 - (target_shape[1] // 2)
            starty = y // 2 - (target_shape[0] // 2)
            frame = frame[
                starty : starty + target_shape[0], startx : startx + target_shape[1]
            ]

        # Paint the mini-map and add the frame to video
        frame[: map_frame_shape[0], : map_frame_shape[1], :] = map_frame
        writer.append_data(frame)

    # After work
    if save_map_positions is not None:
        map_file.close()
    writer.close()
