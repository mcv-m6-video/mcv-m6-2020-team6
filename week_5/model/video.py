import random

import numpy as np

from model.bbox import BBox
from .frame import Frame


class Video:
    list_frames: list

    def __init__(self, list_frames=[]):
        self.list_frames = list_frames

    @staticmethod
    def getgroundtruth(directory_txt, num_frames):
        """Read txt files containing bounding boxes (ground truth and detections)."""
        # Read GT detections from txt file
        # Each value of each line is  "frame_id, x, y, width, height,confidence" respectively
        vid_fr = []
        frameid_saved = 1
        Boxes_list = []
        txt_gt = open(directory_txt, "r")
        for line in txt_gt:
            splitLine = line.split(",")
            frameid = int(splitLine[0])
            label = float(splitLine[1])
            topleft = [float(splitLine[2]), float(splitLine[3])]
            width = float(splitLine[4])
            height = float(splitLine[5])
            confidence = float(splitLine[6])
            id_det = int(splitLine[7])
            Boxes_list.append(
                BBox(frameid, label, topleft, width, height, confidence, id_det)
            )

        for i in range(0, num_frames):
            items = [item for item in Boxes_list if item.frame_id == i]
            if items:
                vid_fr.append(Frame(i, items))

        txt_gt.close()
        return vid_fr

    @staticmethod
    def getgt(directory_txt, ini_frames, end_frames):
        """Read txt files containing bounding boxes (ground truth and detections)."""
        # Read GT detections from txt file
        # Each value of each line is  "frame_id, x, y, width, height,confidence" respectively
        vid_fr = []
        frameid_saved = 1
        Boxes_list = []
        txt_gt = open(directory_txt, "r")
        for line in txt_gt:
            splitLine = line.split(",")
            frameid = int(splitLine[0])
            label = float(splitLine[7])
            topleft = [float(splitLine[2]), float(splitLine[3])]
            width = float(splitLine[4])
            height = float(splitLine[5])
            confidence = float(splitLine[6])
            det_id = int(splitLine[1])
            Boxes_list.append(
                BBox(frameid, label, topleft, width, height, confidence, det_id)
            )

        for i in range(ini_frames, end_frames):
            items = [item for item in Boxes_list if item.frame_id == i]
            if items:
                vid_fr.append(Frame(i, items))
        txt_gt.close()
        return vid_fr

    @staticmethod
    def getgt_detections(directory_txt, ini_frames, end_frames):
        """Read txt files containing bounding boxes (ground truth and detections)."""
        # Read GT detections from txt file
        # Each value of each line is  "frame_id, x, y, width, height,confidence" respectively
        vid_fr = []
        frameid_saved = 1
        Boxes_list = []
        txt_gt = open(directory_txt, "r")
        for line in txt_gt:
            splitLine = line.split(",")
            frameid = int(splitLine[0])
            label = float(splitLine[7])
            topleft = [float(splitLine[2]), float(splitLine[3])]
            width = float(splitLine[4])
            height = float(splitLine[5])
            confidence = float(splitLine[6])
            det_id = int(splitLine[1])
            Boxes_list.append(
                BBox(frameid, label, topleft, width, height, confidence, det_id)
            )

        for i in range(ini_frames, end_frames):
            items = [item for item in Boxes_list if item.frame_id == i]
            if items:
                vid_fr.append(Frame(i, items))
        txt_gt.close()
        return vid_fr

    def modify_random_bboxes(self, prob):
        listbbox = self.list_frames
        rd = np.random.choice([0, 1], size=(len(listbbox),), p=[1 - prob, prob])
        for i, j in enumerate(rd):
            if rd[i]:
                noise = random.uniform(0, 0.5)
                listbbox[i].modify_bbox_frame(noise)
        return listbbox

    def eliminate_random_bboxes(self, prob):
        listbbox = self.list_frames
        rd = np.random.choice([0, 1], size=(len(listbbox),), p=[1 - prob, prob])
        ind = []
        for i, j in enumerate(rd):
            if rd[i]:
                ind.append(i)
        for i in reversed(ind):
            listbbox.pop(i)
        return listbbox

    def get_num_frames(self):
        num_frames = len(self.list_frames)
        return num_frames

    def get_frame_by_id(self, id):
        frame_r = Frame()
        for i in self.list_frames:

            if i.frame_id == id:
                frame_r = Frame(i.frame_id, i.bboxes)

        return frame_r

    def get_detections_all(self):
        listbbox = []
        j = 0
        for i in self.list_frames:
            for j in i.bboxes:
                listbbox.append(j)
        return listbbox

    @staticmethod
    def track2video_kalman(tracking, ini_frames, end_frames):

        list_fr = []
        Boxes_list = []
        for track in tracking:
            for j in track:
                if j:
                    Boxes_list.append(BBox(j[5], -1, [j[1], j[2]], j[3], j[4], 1, j[0]))

        for i in range(ini_frames, end_frames):
            items = [item for item in Boxes_list if item.frame_id == i]
            if items:
                list_fr.append(Frame(i, items))
        vid_fr = Video(list_fr)
        return vid_fr

    @staticmethod
    def track2video_overlap(tracking, ini_frames, end_frames):
        list_fr = []
        Boxes_list = []
        for track in tracking:
            for j in track:
                Boxes_list.append(BBox(j[6], 1, [j[1], j[2]], j[3], j[4], 1, j[0]))

        for i in range(ini_frames, end_frames):
            items = [item for item in Boxes_list if item.frame_id == i]
            if items:
                list_fr.append(Frame(i, items))
        vid_fr = Video(list_fr)
        return vid_fr

    def get_num_tracks(self):
        tracks = []
        for frames in self.list_frames:
            for bbox in frames.bboxes:
                if bbox.det_id not in tracks:
                    tracks.append(bbox.det_id)

        return tracks

    def totrack(self):
        tracking = []
        tracks = []
        for i, j in enumerate(self.list_frames):
            for p in j.bboxes:
                tracks.append(
                    [
                        p.det_id,
                        p.top_left[0],
                        p.top_left[1],
                        p.width,
                        p.height,
                        p.frame_id,
                    ]
                )

            tracking.append(tracks)
            tracks = []
        return tracking

    def get_by_trackid(self, trackid):
        track = []
        for frame in self.list_frames:
            for bbox in frame.bboxes:
                if bbox.det_id == trackid:
                    track.append(bbox)

        return track

    @staticmethod
    def listbboxes2video(listbboxes, num_frames):
        list_fr = []
        for i in range(0, num_frames):
            items = [item for item in listbboxes if item.frame_id == i]
            if items:
                list_fr.append(Frame(i, items))
        vid_fr = Video(list_fr)
        return vid_fr

    def change_track(self, corr, num_frames):
        """Change the id of the tracks selected
        input: corr list containing [track detected, previous track]"""
        newbboxes = []

        for i, j in corr:
            for frame in self.list_frames:
                for bbox in frame.bboxes:
                    if bbox.det_id == j:
                        bbox.det_id = i
                        newbboxes.append(bbox)
        new_tracking_video = Video().listbboxes2video(newbboxes, num_frames)
        return new_tracking_video
