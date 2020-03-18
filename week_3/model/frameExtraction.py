import os
import cv2

from evaluation_funcs import iou_TFTN_video, performance_evaluation, iou_overtime
from model.video import *
from model import bbox
from metrics import *
# need to be installed "brew install ffmpeg"

source = '/Users/claudiabacaperez/Desktop/mcv-m6-2019-team2/datasets/train/S03/c010/'

video_source = source + 'vdo.avi'

folder_frame ='/Users/claudiabacaperez/Desktop/mcv-m6-2019-team2/datasets/train/S03/c010/video_frame'

dir_gt='/Users/claudiabacaperez/Desktop/mcv-m6-2019-team2/datasets/train/S03/c010/det/det_yolo3.txt'
dir_gt='/Users/claudiabacaperez/Desktop/mcv-m6-2019-team2/datasets/train/S03/c010/det/det_yolo3.txt'




video=Video(Video().getgroundTruth(dir_gt,2141))
print('hola')
video.modify_random_bboxes(0.2)

video.eliminate_random_bboxes(0.4)
vido=Video(Video().getgroundTruth(dir_gt,2141))
print('hola')



TP,FP,FN =iou_TFTN_video(vido,video)
[precision, sensitivity, accuracy]= performance_evaluation(TP, FN, FP)


iou_overtime(vido,video, thres=0.1)


print(TP)
print(FP)
print(FN)
print('Precision:',precision)
print('sensitivity:',sensitivity)
print('accuracy:',accuracy)

"""


bb=Video().getgroundTruth()
bb.getgroundTruth(dir_gt)
#print(len(bb.listGd))
bb.modify_random_gt(0.2)
bb.eliminate_random_gt(0.2)
print(len(bb.listGd))
bb2=BoundingBoxes_Video()
bb2.getgroundTruth(dir_gt)
#print(len(bb2.listGd))

list=iou_overtime(bb,bb2)
print(len(list))
[TP,FP,FN]=iou_gt(bb2,bb)
[precision, sensitivity, accuracy]= performance_evaluation(TP, FN, FP)
print(TP)
print(FP)
print(FN)
print('Precision:',precision)
print('sensitivity:',sensitivity)
print('accuracy:',accuracy)"""


