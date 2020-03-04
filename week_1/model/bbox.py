from typing import Optional
import numpy as np
import random


class BBox:
    frame_id: int
    top_left: (float, float)
    width: float
    height: float
    confidence: float

    def __init__(self, frame_id=0, top_left=(0, 0), width=0, height=0, confidence=0):
        self.frame_id = frame_id
        self.top_left = top_left
        self.width = width
        self.height = height
        self.confidence = confidence

    def get_bottom_right(self) -> (float, float):
        return self.top_left[0] + self.width, self.top_left[1] + self.height

    def get_bottom_left(self) -> (float, float):
        return self.top_left[0] + self.width, self.top_left[1]

    def get_top_right(self) -> (float, float):
        return self.top_left[0], self.top_left[1] + self.height

    def get_confidence(self):
        return self.confidence

    def get_frame_id(self):
        return self.frame_id

    def contains_point(self, point: (float, float)) -> bool:
        return (self.top_left[0] <= point[0] <= self.get_bottom_right()[0] and
                self.top_left[1] <= point[1] <= self.get_bottom_right()[1])

    def get_area(self):
        return self.width * self.height

    def union(self, other: 'BBox') -> 'BBox':
        rec = BBox()
        rec.top_left = (min(self.top_left[0], other.top_left[0]), min(self.top_left[1], other.top_left[1]))
        bottom_right = (max(self.get_bottom_right()[0], other.get_bottom_right()[0]),
                        max(self.get_bottom_right()[1], other.get_bottom_right()[1]))

        rec.width = (bottom_right[1] - self.top_left[1]) + 1
        rec.height = (bottom_right[0] - self.top_left[0]) + 1

        return rec

    def intersection(self, other: 'BBox') -> Optional['BBox']:
        rec = BBox()
        if self.contains_point(other.top_left):
            rec.top_left = other.top_left
            rec.height = (other.top_left[0] - self.get_bottom_right()[0]) + 1
            rec.width = (other.top_left[1] - self.get_bottom_right()[1]) + 1
        elif other.contains_point(self.top_left):
            rec.top_left = self.top_left
            rec.height = (self.top_left[0] - other.get_bottom_right()[0]) + 1
            rec.width = (self.top_left[1] - other.get_bottom_right()[1]) + 1
        return rec

    def iou(self, other: 'BBox') -> float:
        return self.intersection(other).get_area() / self.union(other).get_area()

    def to_result(self):
        return [self.top_left[0], self.top_left[1], self.get_bottom_right()[0], self.get_bottom_right()[1]]

    def modify_bbox(self, noise):
        x = random.uniform(self.top_left[0] - self.top_left[1] * noise, self.top_left[0] + self.top_left[0] * noise)
        y = random.uniform(self.top_left[1] - self.top_left[1] * noise, self.top_left[1] + self.top_left[1] * noise)
        self.top_left = [x, y]
        self.width = random.uniform(self.width - self.width * noise, self.width + self.width * noise)
        self.height = random.uniform(self.height - self.height * noise, self.height + self.height * noise)

    def __str__(self):
        return str(self.top_left) + ', ' + str(self.width) + 'x' + str(self.height)
