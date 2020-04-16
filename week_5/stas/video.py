from itertools import chain, islice, repeat
from numbers import Number
from pathlib import Path
from random import choice as rchoice
import sys

import cv2
import imageio
import motmetrics as mm
import numpy as np
from tqdm.auto import tqdm

from bbox import BBox
from colors import all_shades, colors_panetone
from kalman_filter_gps import TrackList
from reader import Video_Detection
from utils import (
    draw_detections,
    match_detections,
    pixel2coords,
    undistort_points,
    get_timestamp,
)


class Video:
    """Allows to iterate over frames, detection bounding-boxes, tracking bounding-boxes
    and mini-map positions.

    Video uses a concept of Methods for detection and tracking.
    Method is a Callable that accepts as params:
        1) frame_id - If you already have cached results per frame
        2) frame_id, frame - [Detection] If you want to compute BBoxes on the fly
        3) frame_id, detections, frame - [Tracking] If you want to track on the fly
    Method should return an Iterable of BBoxes. Tracking methods should set
    BBox.obj_id.

    Attributes:
        path (Path): path to the .avi/.mp4 file
        frames (imageio.reader): access to raw (possibly distorted) frames
        camera_num (int): the number of the camera
        sequence_num (int): the number of the sequence
        gt (Video_Detection): tracking ground truth
        distortion (bool): Whether raw frames are distorted
        gpsH (np.array): Homography to convert pixels to gps coords
        camera_matrix (np.array, optional): Intrisic paramters of the camera
            if video is distorted
        distortion_coeffs(np.array, optional): Distortion coefficients
        verbose (bool): Whether to print information while working
    """

    def __init__(self, path, detection_method=None, tracking_method=None, verbose=True):
        """Initialize the video.

        Args:
            path (str or Path): path to the .avi/.mp4 file
            detection_method (Method, optional): The default detection method to use.
                To understand methods see Video.__doc_str__.
            tracking_method (Method, optional): The default tracking method to use.
            verbose (bool, optional): Whether to print usefull info.
        """
        assert Path(path).exists(), f"File {path} does not exist"
        self.path = Path(path)
        self.frames = imageio.get_reader(path)
        self.camera_num = int(path.parent.name[1:])
        self.sequence_num = int(path.parent.parent.name[1:])
        self.start_time = get_timestamp(path)
        self.gt = Video_Detection(path.parent / "gt" / "gt.txt", total_frames=len(self))
        self.init_calibration(path.parent / "calibration.txt")
        self.detection_method = detection_method
        self.tracking_method = tracking_method
        self.verbose = verbose

    def init_calibration(self, calibration_path):
        """Read and store camera calibration data"""

        def extract_matrix(line):
            line = line.split(":", 1)[-1]
            rows = line.count(";") + 1
            line = line.replace(";", " ")
            return np.fromstring(line, sep=" ").reshape(rows, -1)

        with open(calibration_path, "rt") as calib:
            for line in calib:
                matrix = extract_matrix(line)
                if line.startswith("Homography matrix:"):
                    self.gpsH = np.linalg.inv(matrix)
                elif line.startswith("Intrinsic parameter matrix:"):
                    self.camera_matrix = matrix
                elif line.startswith("Distortion coefficients:"):
                    self.distortion_coeffs = matrix
        self.distortion = hasattr(self, "camera_matrix") and hasattr(
            self, "distortion_coeffs"
        )

    def print(self, *args):
        """Print only if set to verbose"""
        if self.verbose:
            print(*args)

    def tqdm(self, iterable, **kwargs):
        """Use tqdm (1) on stdout (2) only if verbose"""
        if self.verbose:
            if "file" not in kwargs:
                kwargs["file"] = sys.stdout
            return tqdm(iterable, **kwargs)
        return iterable

    def __iter__(self):
        """Return iterator over undistorted frames"""
        if not self.distortion:
            yield from self.frames
            return

        for frame in self.frames:
            yield cv2.undistort(frame, self.camera_matrix, self.distortion_coeffs)

    def __len__(self):
        return self.frames.count_frames()

    @property
    def fps(self):
        """Frames Per Second"""
        if not hasattr(self, "_fps"):
            self._fps = self.frames.get_meta_data()["fps"]
        return self._fps

    def smart_undistort_bboxes(self, bboxes, enable=True):
        if enable and self.distortion:
            return map(lambda b: self.undistort_bbox(b), bboxes)
        return bboxes

    def detect(self, method=None, undistort_bboxes=True):
        """Output detection bboxes on this video.

        Args:
            method (Method, optional): Use method. If not supplied
                switch to self.detection_method
            undistort_boxes (bool): Whether to apply undistortion to bounding boxes

        """
        method = method or self.detection_method
        cheat_fast = True
        try:
            method(0)
        except TypeError:
            cheat_fast = False
        if not cheat_fast:
            for frame_id, frame in enumerate(self):
                yield self.smart_undistort_bboxes(
                    method(frame_id, frame), undistort_bboxes
                )
        else:
            for frame_id in range(len(self)):
                yield self.smart_undistort_bboxes(method(frame_id), undistort_bboxes)

    def tracking(self, method=None, detection_method=None, undistort_bboxes=True):
        method = method or self.tracking_method
        detection_method = detection_method or self.detection_method

        cheat_fast = True
        try:
            method(0)
        except TypeError:
            cheat_fast = False

        if not cheat_fast:
            for frame_id, (detections, frame) in enumerate(
                zip(self.detect(detection_method, undistort_bboxes), self)
            ):
                yield self.smart_undistort_bboxes(
                    method(frame_id, detections, frame), undistort_bboxes
                )
        else:
            for frame_id in range(len(self)):
                yield self.smart_undistort_bboxes(method(frame_id), undistort_bboxes)

    def eval_detection(
        self, method=None, iou_thresholds=tuple(map(lambda x: x / 100, range(50, 96, 5)))
    ):
        """Evaluate detection method.

        Args:
            method (Method): Method to evaluate
            iou_thresholds (tuple[float], optional): IOU_thresholds to evaluate.

        Returns:
            float: mAP score
        """
        if isinstance(iou_thresholds, Number):
            return self.eval_detection_at(method, iou_thresholds)
        APsum, APcount = 0, 0
        for thres in iou_thresholds:
            APsum += self.eval_detection_at(method, thres)
            APcount += 1
        mAP = APsum / APcount
        self.print(f"Overall mAP: {mAP}")
        return mAP

    def eval_detection_at(self, method=None, iou_threshold=0.1):
        """Evaluate detection method at single iou_threshold"""
        self.print(f"Evaluating detections @{iou_threshold}")
        with self.gt as gt:
            # TODO: check if self.total_frames is working
            # gt = chain(gt, repeat(iter(())))
            gt = self.tqdm(gt, total=len(self))
            matches = (
                match_detections(detections, gt_boxes, iou_threshold)
                for detections, gt_boxes in zip(self.detect(method), gt)
            )
            matches = chain.from_iterable(matches)
            matches = sorted(matches, key=lambda m: m[0].confidence)
            TP = np.fromiter(map(lambda x: x[1] is not None, matches), bool)
        precision = TP.cumsum() / (np.arange(len(TP)) + 1)
        precision = np.flip(np.maximum.accumulate(precision[::-1]))

        recall = TP.cumsum() / len(self.gt)
        recall_diff = np.diff(np.insert(recall, 0, 0))
        score = (precision * recall_diff).sum()
        self.print(f"AP@{iou_threshold}: {score}")
        return score

    def visualize(self, *methods, begin=0, end=100, method_names=None):
        """Visualize detection results

        Args:
            *methods (Union[Method, Iterable[Method]]): Method(s) to visualize
            begin (int, optional): starting frame
            end (int, optional): ending frame
            method_names (Iterable[str], optional): If supplied must be
                same length as the number of methods. Causes bounding boxes
                to be signed with the method name.

        Yields:
            np.array: Frame with painted BBoxes
        """

        frames = islice(self.frames, begin, end)
        for frame, frame_id in zip(frames, range(begin, end)):
            for i, (method, colors) in enumerate(zip(methods, all_shades)):
                # TODO should allow calling differently
                boxes = method(frame_id, frame)
                txt = method_names[i] if method_names else None
                frame = draw_detections(
                    frame, boxes, txt, colors=colors, text_color=colors[0]
                )
            yield frame

    def dump_visualization(
        self, save_path, method, begin=0, end=100, fps=None, with_gt=False
    ):
        """Save detection visualization to path"""
        self.print(f"Dumping visualization to {save_path}")
        if fps is None:
            fps = self.frames.get_meta_data()["fps"]
        with self.gt as gt:
            method = (method, gt) if with_gt else (method,)
            names = ["TEST", "GT"] if with_gt else None
            writer = imageio.get_writer(save_path, fps=fps)
            for frame in self.tqdm(
                self.visualize(*method, begin=begin, end=end, method_names=names),
                total=(end - begin),
            ):
                writer.append_data(frame)
        writer.close()

    def eval_tracking(self, method, detection_method=None, iou_threshold=0.5):
        """Evaluate tracking method using IDF1"""

        def extract(boxes):
            boxes = list(boxes)
            objs = list(map(lambda box: box.obj_id, boxes))
            box_arr = np.stack([box.ltwh for box in boxes]) if boxes else np.array([])
            return objs, box_arr

        self.print(f"Evaluating tracking...")
        accumulator = mm.MOTAccumulator(auto_id=True)

        with self.gt as gt:
            gt = chain(gt, repeat(iter(())))
            gt = self.tqdm(gt, total=len(self))
            for tracks, gt_boxes in zip(
                self.tracking(method, detection_method, False), gt
            ):
                gt_objs, gt_box_arr = extract(gt_boxes)
                track_objs, track_box_arr = extract(tracks)
                dists = mm.distances.iou_matrix(
                    gt_box_arr, track_box_arr, max_iou=iou_threshold
                )

                accumulator.update(
                    gt_objs, track_objs, dists,
                )

        mh = mm.metrics.create()
        summary = mh.compute(
            accumulator, metrics=["num_frames", "idf1", "mota"], name="Full"
        )

        self.print(summary)
        return summary["idf1"][0]

    def gps2map_frame(self, gps, pix_per_10_meters=50, map_frame_shape=(400, 400)):
        """Map GPS coordinates to mini-map coordinate"""
        # FormerTODO: make gps2map_frame based on stats not fixed values
        # NO! Drawing based on fixed values makes the locations stable across cameras

        x, y = gps
        # Centers from ReadMe.txt
        sequence_centers = [
            None,
            (42.525678, -90.723601),
            (42.491916, -90.723723),
            (42.498780, -90.686393),
            (42.498780, -90.686393),
            (42.498780, -90.686393),
        ]
        cet_x, cet_y = sequence_centers[self.sequence_num]
        # Relative to the center of location
        x, y = x - cet_x, y - cet_y
        # 1e-4 degrees is around 10 meters in real world
        div = 1e-4
        x, y = x / div, y / div
        x, y = x * pix_per_10_meters, y * pix_per_10_meters
        # Relative to corner 0,0 of map_frame
        x, y = x + map_frame_shape[1] / 2, y + map_frame_shape[0] / 2
        # print(f"{gps} -> {int(x), int(y)}")
        return int(x), int(y)

    def undistort_point(self, p):
        """Compute the undistortet version of coords p"""
        if not self.distortion:
            return p
        p = np.array(p).reshape(-1, 2)
        p = undistort_points(p, self.camera_matrix, self.distortion_coeffs)[0]
        return tuple(p[:2])

    def undistort_bbox(self, bbox):
        if not self.distortion:
            return bbox
        return bbox.undistorted(self.camera_matrix, self.distortion_coeffs)

    def box2gps(self, b: BBox):
        """Reduce BBox to single gps position"""
        p = b.ground_pos()
        if not b.is_undistorted:
            p = self.undistort_point(p)
        p = pixel2coords(p, self.gpsH)
        return p

    def gps_positions(self, with_ids=True):
        """Iterator over all gps positions"""
        bboxes_source = self.tracking() if with_ids else self.detect()
        for bboxes in bboxes_source:
            bboxes = list(bboxes)
            positions = map(lambda b: self.box2gps(b), bboxes)
            if not with_ids:
                yield positions
            else:
                ids = map(lambda b: b.obj_id, bboxes)
                yield zip(positions, ids)

    def map_positions(
        self, with_ids=True, pix_per_10_meters=50, map_frame_shape=(400, 400)
    ):
        """Iterator over all mini-map positions"""
        to_map = lambda pos: self.gps2map_frame(  # noqa
            pos, pix_per_10_meters, map_frame_shape
        )
        for gps_set in self.gps_positions(with_ids):
            if with_ids:
                yield map(
                    lambda pos_id: (to_map(pos_id[0]), pos_id[1],), gps_set,
                )
            else:
                yield map(to_map, gps_set)

        for bboxes in self.tracking():
            bboxes = list(bboxes)
            positions = map(lambda b: self.box2gps(b), bboxes)
            map_positions = map(
                lambda pos: self.gps2map_frame(pos, pix_per_10_meters, map_frame_shape),
                positions,
            )
            if not with_ids:
                yield map_positions
            else:
                ids = map(lambda b: b.obj_id, bboxes)
                yield zip(map_positions, ids)

    def dump_visualized_map(
        self, save_path, begin, end, fps=10, lasting=False, map_frame_shape=(400, 400)
    ):
        """Make and save video file with live tracks on mini-map"""
        self.print("Visualizing GPS locations..")
        frames = islice(self, begin, end)
        frames = self.tqdm(frames, file=sys.stdout, total=end - begin)

        bboxes_sets = islice(self.tracking(), begin, end)
        map_position_sets = islice(self.map_positions(with_ids=False), begin, end)

        writer = imageio.get_writer(save_path, fps=fps)

        map_frame = np.ones(map_frame_shape + (3,)) * 255
        colors = all_shades[0]

        for frame, map_positions, bboxes in zip(frames, map_position_sets, bboxes_sets):
            bboxes = list(bboxes)
            frame = draw_detections(frame, bboxes, colors=colors)
            for map_position, box in zip(map_positions, bboxes):
                if isinstance(box.obj_id, Number) and box.obj_id != -1:
                    color = colors[box.obj_id % len(colors)]
                else:
                    color = rchoice(colors_panetone)
                if not lasting:
                    map_frame = np.ones(map_frame_shape + (3,)) * 255
                cv2.drawMarker(map_frame, map_position, color, thickness=4)
                gx, gy = box.ground_pos()
                cv2.drawMarker(frame, (int(gx), int(gy)), color, thickness=4)

            frame[: map_frame_shape[0], 0 : map_frame_shape[1], :] = map_frame
            writer.append_data(frame)

        writer.close()

    def frame_ids(self):
        return iter(range(len(self)))


class _VideoSetIterator:
    """Class for iterating over time and multiple cameras"""

    def __init__(self, times):
        self.times = times

    def __iter__(self):
        return self

    def __next__(self):
        if not self.times:
            self.track_map_on = False
            raise StopIteration

        next_iter, v = min(self.times, key=self.times.get)
        try:
            next_data = next(next_iter)
            self.times[next_iter, v] += 1 / v.fps
            return next_data
        except StopIteration:
            del self.times[next_iter, v]
        return self.__next__()


class VideoSet:
    """Provides chronological iterator over many cameras over:
        - frames
        - detection bboxes
        - tracking bboxes
        - mini-map positions
    """

    def __init__(self, videos=tuple()):
        self.videos = tuple(videos)

    def __iter__(self):
        times = {(iter(v), v): v.start_time for v in self.videos}
        return _VideoSetIterator(times)

    def __len__(self):
        return sum(len(v) for v in self.videos)

    def detect(self):
        times = {(v.detect(), v): v.start_time for v in self.videos}
        return _VideoSetIterator(times)

    def tracking(self):
        times = {(v.tracking(), v): v.start_time for v in self.videos}
        return _VideoSetIterator(times)

    def map_positions(
        self, with_ids=True, pix_per_10_meters=50, map_frame_shape=(400, 400)
    ):
        times = {
            (
                v.map_positions(with_ids, pix_per_10_meters, map_frame_shape),
                v,
            ): v.start_time
            for v in self.videos
        }
        return _VideoSetIterator(times)

    def gps_positions(self, with_ids=True):
        times = {(v.gps_positions(with_ids), v): v.start_time for v in self.videos}
        return _VideoSetIterator(times)

    def timestamps(self):
        def frame_times(vid):
            return map(lambda i: vid.start_time + i / vid.fps, range(len(vid)))

        times = {(frame_times(v), v): v.start_time for v in self.videos}
        return _VideoSetIterator(times)

    def video_objs(self):
        times = {(repeat(v, len(v)), v): v.start_time for v in self.videos}
        return _VideoSetIterator(times)

    def frame_ids(self):
        times = {(v.frame_ids(), v): v.start_time for v in self.videos}
        return _VideoSetIterator(times)

    def gts(self):
        times = {(iter(v.gt), v): v.start_time for v in self.videos}
        return _VideoSetIterator(times)

    def multi_camera_tracking(self, use_encoding):
        track_list = TrackList(encoding=use_encoding)
        for gps, vid, timestamp, detections, image in zip(
            self.gps_positions(with_ids=True),
            self.video_objs(),
            self.timestamps(),
            self.tracking(),
            self,
        ):
            gps, bboxes = list(gps), list(detections)
            if use_encoding:
                camera_tracks = []
                for gps_track, bbox in zip(gps, detections):
                    camera_tracks.append([gps_track[0], gps_track[1], bbox])
                id_map, gps_global_ids = track_list.process_detections(
                    detections=camera_tracks, time=timestamp, image=image
                )
            else:
                id_map, _ = track_list.process_detections(
                    detections=gps, time=timestamp
                )
            for bbox in bboxes:
                bbox.obj_id = id_map[bbox.obj_id]
            yield bboxes

    def eval_tracking(self, iou_threshold=1, use_encoding=True):
        """Evaluate tracking method using IDF1"""

        def extract(boxes):
            boxes = list(boxes)
            objs = list(map(lambda box: box.obj_id, boxes))
            box_arr = np.stack([box.ltwh for box in boxes]) if boxes else np.array([])
            return objs, box_arr

        accumulator = mm.MOTAccumulator(auto_id=True)

        gt = tqdm(self.gts(), total=len(self))
        for tracks, gt_boxes in zip(self.multi_camera_tracking(use_encoding), gt):
            gt_objs, gt_box_arr = extract(gt_boxes)
            track_objs, track_box_arr = extract(tracks)
            dists = mm.distances.iou_matrix(
                gt_box_arr, track_box_arr, max_iou=iou_threshold
            )

            accumulator.update(
                gt_objs, track_objs, dists,
            )

        mh = mm.metrics.create()
        summary = mh.compute(
            accumulator,
            metrics=["num_frames", "idf1", "mota", "idp", "idr", "precision", "recall"],
            name="Full",
        )

        print(summary)
        return summary["idf1"][0]
