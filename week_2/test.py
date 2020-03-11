import os
import re
import glob
import cv2
import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from tqdm import tqdm
from backgroundSubstration import *


frames_dir = '../datasets/train/S03/c010/frames/'
roi_path = '../datasets/train/S03/c010/roi.jpg'
gt_dir = 'annotation.txt'

train_list, test_list=get_frames(frames_dir, trainbackground=0.25)

image_list_fg=fg_mask_single_gaussian(frames_dir,roi_path)


#mask,listbboxes=connected_component(image_list_fg[18],min_area=1200)
listbboxes,listofmask,video_gt=connected_component_test(image_list_fg,min_area=1500)



gt_video = Video(Video().getgroundTruthown(gt_dir, 0, 2141))
gt_video2=get_video_frameid(gt_video,26)
detections = Video(video_gt)
detections2=get_video_frameid(detections,26)

print('done')