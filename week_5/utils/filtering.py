from collections import Counter

import cv2
from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import distance

from model.video import Video
from model.bbox import BBox


def dif(t, bboxes_r):
    distmin = np.inf
    for br in bboxes_r:
        dist = distance.euclidean(t, br)
        if distmin > dist:
            distmin = distance.euclidean(t, br)

    return distmin


def preprocess_videodetections(video_det, fin_frame, roi_path):
    """
    preprocess_videodetections(video_det,gt_dir)
    The parked cars are erased from the detections
    :param video_det:
    :param gt_dir:
    :return: video
    """
    listofbboxes = video_det.get_detections_all()
    listoftopleft = []

    for i in listofbboxes:
        listoftopleft.append(i.top_left)

    bboxes_r = list(
        [k for k, v in Counter(map(tuple, listoftopleft)).items() if v > 10]
    )
    """for j in range(1,len(listframes)-2):
        for i in range(j,j+20):"""

    bboxes = list([list(i) for i in bboxes_r])

    new_bb = []

    for track in listofbboxes:
        distmin = dif(track.top_left, bboxes)
        if distmin > 10:
            new_bb.append(track)

    video = Video().listbboxes2video(new_bb, fin_frame)

    video = roi_detections(video, roi_path, fin_frame)
    return video


def roi_detections(video_det, roi_path, fin_frame):
    """
    roi_detections(video_det,path_roi)
    Delete detections outside the roi
    :param video_det:
    :param path_roi:
    :return: video
    """
    gray = cv2.imread(roi_path)  # roi image
    gray_n = 1 * (gray[:, :, 0] > 0)
    list_bbox = []
    height, width = gray_n.shape
    gray_n[0:int(0.15 * height), 0:width] = 0
    for i in video_det.get_detections_all():
        bbox_in = gray_n[int(i.top_left[1]):int(i.top_left[1] + i.height),
                  int(i.top_left[0]):int(i.top_left[0] + i.width)]
        if (np.mean(bbox_in) >= 0.6):
            list_bbox.append(i)

    video = Video().listbboxes2video(list_bbox, fin_frame)

    return video


def preprocess_tracking(tracking_list):
    """
    preprocess_annotations(gt_dir)
    The parked cars are erased from the annotations
    :param gt_dir:
    :return:
    """
    listoftracks = tracking_list
    listoftopleft = []
    for track in listoftracks:
        for j in track:
            listoftopleft.append([j[1], j[2]])

    bboxes_r = list(
        [k for k, v in Counter(map(tuple, listoftopleft)).items() if v > 500]
    )
    trackids = []
    """bboxes = list([list(i) for i in bboxes_r])
    p = [listoftracks[0][0][1], listoftracks[0][0][2]]
    for i in listoftracks:
        for box in bboxes_r:
            u=i[0][1]
            pu=i[0][2]
            if( i[0][1]==box[0] and i[0][2]==box[1]):
                trackids.append(i[0][6])"""

    [trackids.append(i[0][0]) for i in listoftracks if ([i[0][1], i[0][2]] in bboxes_r)]
    for track in tracking_list:
        for j in track:
            distmin = dif([j[1], j[2]], bboxes_r)
            if distmin < 300:
                trackids.append(j[0])
    trackids = set(trackids)

    for track in tracking_list:
        for j in track:
            for id in trackids:
                if j[0] == id:
                    track.remove(j)


def visualize_bb(detections_video, begin, end, images):
    num_frames = end - begin

    fig, ax1 = plt.subplots(1)

    fig.suptitle("Foreground detection")

    plt.ion()

    for i in range(num_frames):

        detections_bboxes = detections_video.get_frame_by_id(begin + i)
        ax1.set_title("Detection")
        ax1.imshow(cv2.imread(images[i]))

        for bbox in detections_bboxes.bboxes:
            detec = patches.Rectangle(
                bbox.top_left,
                bbox.width,
                bbox.height,
                linewidth=1.5,
                edgecolor="r",
                facecolor="none",
                label="detections",
            )

            ax1.add_patch(detec)
        ax1.axis("off")

        plt.show()
        plt.pause(0.005)
        ax1.clear()

def iou(bboxA: BBox, bboxB: BBox):
    # compute the intersection over union of two bboxes

    # Format of the bboxes is [tly, tlx, bry, brx, ...], where tl and br
    # indicate top-left and bottom-right corners of the bbox respectively.

    # determine the coordinates of the intersection rectangle
    xA = max(bboxA.top_left[0], bboxB.top_left[0])
    yA = max(bboxA.top_left[1], bboxB.top_left[1])
    xB = min(bboxA.get_bottom_right()[0], bboxB.get_bottom_right()[0])
    yB = min(bboxA.get_bottom_right()[1], bboxB.get_bottom_right()[1])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both bboxes
    bboxAArea = (bboxA.get_bottom_right()[1] - bboxA.top_left[1] + 1) * (
            bboxA.get_bottom_right()[0] - bboxA.top_left[0] + 1
    )
    bboxBArea = (bboxB.get_bottom_right()[1] - bboxB.top_left[1] + 1) * (
            bboxB.get_bottom_right()[0] - bboxB.top_left[0] + 1
    )

    iou = interArea / float(bboxAArea + bboxBArea - interArea)

    # return the intersection over union value
    return iou
