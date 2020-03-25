import numpy as np
import cv2

img1 = '../datasets/kitti/images/000045_10.png'
img2 = '../datasets/kitti/images/000045_11.png'
frame1 = (cv2.imread(img1, cv2.IMREAD_GRAYSCALE))
frame2 = (cv2.imread(img2, cv2.IMREAD_GRAYSCALE))

nvof = cv2.cuda_NvidiaOpticalFlow_1_0.create(frame1.shape[1], frame1.shape[0], 5, False, False, False, 1)
flow = nvof.calc(frame1, frame2, None)
flowUpSampled = nvof.upSampler(flow[0], frame1.shape[1], frame1.shape[0], nvof.getGridSize(), None)

cv2.writeOpticalFlow('OpticalFlow.flo', flowUpSampled)

nvof.collectGarbage()