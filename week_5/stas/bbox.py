import numpy as np

from utils import undistort_points


class BBox:
    def __init__(
        self,
        left,
        top,
        width,
        height,
        frame_id=None,
        obj_id=None,
        confidence=1.0,
        hists=None,
        is_undistorted=False,
    ):
        if all(isinstance(i, int) for i in (left, top, width, height)):
            self._ltwhrb = np.array(
                [left, top, width, height, width + left - 1, height + top - 1]
            )
        else:
            self._ltwhrb = np.array(
                [left, top, width, height, width + left, height + top]
            )
        self.frame_id = frame_id
        self.obj_id = obj_id
        self.confidence = confidence
        self.hists = hists
        self.is_undistorted = is_undistorted

    @property
    def ltwh(self):
        return self._ltwhrb[:4]

    @property
    def ltrb(self):
        return self._ltwhrb[[0, 1, 4, 5]]

    @property
    def area(self):
        w, h = self._ltwhrb[2:4]
        return w * h

    def __and__(self, other):
        """Calculate the area of the intersection of self with the other"""
        l, t, r, b = self.ltrb
        rl, rt, rr, rb = other.ltrb
        nl, nt, nr, nb = max(l, rl), max(t, rt), min(r, rr), min(b, rb)
        if nl > nr or nt > nb:
            return 0
        return (nr - nl + 1) * (nb - nt + 1)

    def __or__(self, other):
        """Calculate the area of the union of self with the other"""
        return self.area + other.area - (self & other)

    def __matmul__(self, other):
        """Calculate IOU of self with the other"""
        return (self & other) / (self | other)

    def sort_matches(self, others):
        """Sort others according to iou score with self"""
        return sorted(others, key=lambda box: self @ box, reverse=True)

    def undistorted(self, camera_matrix, distortion_coeffs):
        if self.is_undistorted:
            return self
        pts = self.ltrb.reshape((-1, 2))
        pts = undistort_points(pts, camera_matrix, distortion_coeffs)
        l, t, r, b = pts.reshape(-1)
        return BBox(
            l,
            t,
            r - l,
            b - t,
            frame_id=self.frame_id,
            obj_id=self.obj_id,
            confidence=self.confidence,
            is_undistorted=True,
        )

    def ground_pos(self):
        l, t, r, b = self.ltrb
        bt_coeff = b / 1080
        bt_off = (b - t) / 2 * bt_coeff
        x, y = (l + r) / 2, (b - bt_off)
        return x, y
