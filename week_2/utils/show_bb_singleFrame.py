from metrics.evaluation_funcs import *
from model.frame import *
import matplotlib.pyplot as plt
from matplotlib import patches
import cv2
from PIL import Image
import numpy as np
from metrics import *

def show_bboxes(path, bboxes: Frame, bboxes_noisy: Frame):
    """
    shows the ground truth and the noisy bounding boxes
    :param path:
    :param bboxes:
    :param bboxes_noisy:
    :return:
    """
    im = np.array(Image.open(path), dtype=np.uint8)
    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)
    iouframe, TP, FP, FN= iou_frame(bboxes, bboxes_noisy, thres=0.1)
    # Create a Rectangle patch
    i=0
    for bbox in bboxes.bboxes:
        rect = patches.Rectangle((bbox.top_left),
                                 bbox.width , bbox.height ,
                             linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

    for bbox_noisy in bboxes_noisy.bboxes:
        bb = bbox_noisy.to_result()
        rect = patches.Rectangle((bbox_noisy.top_left),
                                 bbox_noisy.width , bbox_noisy.height,
                             linewidth=1, edgecolor='r', facecolor='none')

        ax.add_patch(rect)
        i+=1
    # Add the patch to the Axes
    plt.axis('off')
    plt.show()



"""
path= 'D:/Documents/Proves/week1/datasets/train/S03/c010/frames/image391.jpg'

gt_dir = 'D:\\Documents\\Proves\\week1\\annotation.txt'
gt_video = Video(Video().getgroundTruthown(gt_dir,391,392))
gt_video_modif1 = Video(Video().getgroundTruthown(gt_dir,391,392))

gt_video_modif1.modify_random_bboxes(0.7)


detections_bboxes = gt_video_modif1.get_frame_by_id(391)
gt_bboxes = gt_video.get_frame_by_id(391)

iouframe, TP,FP,FN=iou_frame(detections_bboxes,gt_bboxes, thres=0.7)

for i in iouframe:
    print('\n IOU:',i)
show_bboxes(path,gt_bboxes,detections_bboxes)
"""


"""
path = 'D:/Documents/Proves/week1/datasets/train/S03/c010/frames/image200.jpg'

gt_dir = 'D:\\Documents\\Proves\\week1\\annotation.txt'
gt_video = Video(Video().getgroundTruthown(gt_dir, 200))

yolo = 'D:/Documents/Proves/week1/datasets/train/S03/c010/det/det_yolo3.txt'
yolo_video = Video(Video().getgroundTruthown(yolo, 200))


gt_bboxes = gt_video.get_frame_by_id(200)
detections_bboxes = yolo_video.get_frame_by_id(200)


iouframe, TP,FP,FN=iou_frame(gt_bboxes, detections_bboxes, thres=0.5)

for i in iouframe:
    print('\n IOU:',i)
show_bboxes(path,gt_bboxes,detections_bboxes)
"""