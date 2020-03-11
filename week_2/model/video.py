from typing import Optional
from .bbox import BBox
from .frame import Frame
import numpy as np
import random


class Video:
    list_frames: list

    def __init__(self, list_frames=[]):
        self.list_frames=list_frames

    @staticmethod
    def getgt_detections(directory_txt,num_frames):
        """Read txt files containing bounding boxes (ground truth and detections)."""
        # Read GT detections from txt file
        # Each value of each line is  "frame_id, x, y, width, height,confidence" respectively
        vid_fr=[]
        frameid_saved = 1
        Boxes_list = []
        txt_gt = open(directory_txt, "r")
        for line in txt_gt:
            splitLine = line.split(",")
            frameid = int(splitLine[0])
            topleft = [float(splitLine[2]), float(splitLine[3])]
            width = float(splitLine[4])
            height = float(splitLine[5])
            confidence = float(splitLine[6])
            Boxes_list.append(BBox(frameid,topleft ,width, height, confidence))

        for i in range(0, num_frames):
            items = [item for item in Boxes_list if item.frame_id ==i]
            if items:
                vid_fr.append(Frame(i,items))

        txt_gt.close()
        return vid_fr

    @staticmethod
    def getgroundTruthown(directory_txt, ini_frames,end_frames):
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
            topleft = [float(splitLine[2]), float(splitLine[3])]
            width = float(splitLine[4])
            height = float(splitLine[5])
            confidence = float(splitLine[6])
            trackid=int(splitLine[7])
            Boxes_list.append(BBox(frameid,trackid,topleft, width, height, confidence))

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
            if (rd[i]):
                noise = random.uniform(0, 0.5)
                listbbox[i].modify_bbox_frame(noise)
        return listbbox

    def eliminate_random_bboxes(self, prob):
        listbbox = self.list_frames
        rd = np.random.choice([0, 1], size=(len(listbbox),), p=[1 - prob, prob])
        ind = []
        for i, j in enumerate(rd):
            if (rd[i]): ind.append(i)
        for i in reversed(ind): listbbox.pop(i)
        return listbbox

    def get_num_frames(self):
        num_frames = len(self.list_frames)
        return num_frames

    def get_frame_by_id(self, id):
        index = []
        frame_r=Frame()
        j = 0
        for i in self.list_frames:

            if i.frame_id == id:
                frame_r = Frame(i.frame_id, i.bboxes)

        return frame_r

    def get_detections_all(self):
        listbbox = []

        for i in self.list_frames:
            for j in i.bboxes:
                listbbox.append(j)
        return listbbox

    def get_detections_by_track(self, track):
        listbbox=[]
        detections=self.get_detections_all()
        [listbbox.append(i) for i in detections if (i.track_id==track)]
        return listbbox