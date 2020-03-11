import os
import re
import glob
import cv2
import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from tqdm import tqdm
from backgroundSubstration import *
from evaluation_funcs import *


frames_dir = '../datasets/train/S03/c010/frames/'
roi_path = '../datasets/train/S03/c010/roi.jpg'

train_list, test_list=get_frames(frames_dir, trainbackground=0.25)

image_list_fg=fg_mask_single_gaussian(frames_dir,roi_path)

