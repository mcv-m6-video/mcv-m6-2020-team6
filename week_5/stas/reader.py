from itertools import groupby
from pathlib import Path
import pickle
from typing import Union

from bbox import BBox


class Video_GT:
    def __init__(self, gt_path: Union[str, Path], total_frames=None):
        assert Path(gt_path).exists(), f"File {gt_path} does not exist"
        self.gt_path = Path(gt_path)
        self.total_frames = total_frames
        self._enter_count = 0

    def __enter__(self):
        if self._enter_count == 0:
            self.gt_file = open(self.gt_path, "rt")
        self._enter_count += 1
        return self

    def __exit__(self, *_):
        self._enter_count -= 1
        if self._enter_count == 0:
            self.gt_file.close()
            delattr(self, "gt_file")

    def __len__(self):
        if not hasattr(self, "_computed_len"):
            self.__enter__()
            for i, _ in enumerate(self.gt_file):
                pass
            self._computed_len = i + 1
            self.__exit__()
        return self._computed_len

    def detections(self):
        assert hasattr(self, "gt_file"), (
            "Wrong use! Correct usage:\n"
            "with Video_GT(..) as gts:\n"
            "    for i in gts:"
        )
        for line in self.gt_file:
            splitted = line.split(",")
            frame_id, obj_id, *bbox = tuple(map(lambda x: int(float(x)), splitted[:6]))
            confidence = float(splitted[6])
            confidence = confidence if confidence != -1 else 1
            yield BBox(
                *bbox, frame_id=frame_id - 1, obj_id=obj_id, confidence=confidence
            )

    def __iter__(self):
        old_frame_id = -1
        for frame_id, detections in groupby(
            self.detections(), key=lambda x: x.frame_id
        ):
            while old_frame_id + 1 != frame_id:
                yield iter(())
                old_frame_id += 1
            yield detections
            old_frame_id = frame_id
        if self.total_frames is None:
            return

        while old_frame_id + 1 < self.total_frames:
            yield (iter(()))
            old_frame_id += 1


class Video_Detection(Video_GT):
    @property
    def detection_dict(self):
        if not hasattr(self, "_detection_dict"):
            self.__enter__()
            it = super().__iter__()
            self._detection_dict = {i: list(dets) for i, dets in enumerate(it)}
            self.__exit__()
        return self._detection_dict

    def __call__(self, frame_id, *_):
        return self.detection_dict.get(frame_id, list())

    def __iter__(self):
        num_iters = self.total_frames or 2 ** 30
        for i in range(num_iters):
            yield self.detection_dict.get(i, [])


class TrackReader:
    def __init__(self, path):
        assert Path(path).exists(), f"Path {path} does not exist!"
        with open(path, "rb") as f:
            track_list = pickle.load(f)

        def track_to_bbox(trl):
            return BBox(*trl[1:5], frame_id=trl[-1], obj_id=trl[0], hists=trl[5])

        frames = set([track[-1] for tracks in track_list for track in tracks])
        self.track_dict = {frame: [] for frame in frames}
        for tracks in track_list:
            for track in tracks:
                self.track_dict[track[-1]].append(track_to_bbox(track))

    def __call__(self, frame_id, *_):
        return self.track_dict.get(frame_id, [])
