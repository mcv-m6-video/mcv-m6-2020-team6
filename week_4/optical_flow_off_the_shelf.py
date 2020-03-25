import cv2
import imageio
import numpy as np
#import pyflow
import time
from week_4.pyoptflow.pyoptflow import HornSchunck, getimgfiles
from week_4.pyoptflow.pyoptflow.plots import compareGraphs
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


def Coarse2Fine(img1, img2, visualize=True):
    image1 = cv2.cvtColor(cv2.imread(img1), cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(cv2.imread(img2), cv2.COLOR_BGR2GRAY)

    im1 = image1[:, :, np.newaxis]
    im2 = image2[:, :, np.newaxis]

    im1 = np.asarray(im1).astype(float) / 255.
    im2 = np.asarray(im2).astype(float) / 255.

    # Flow Options:
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

    s = time.time()
    u, v, im2W = pyflow.coarse2fine_flow(
        im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
    e = time.time()
    print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
        e - s, im1.shape[0], im1.shape[1], im1.shape[2]))
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)

    flow_1 = np.ndarray((u.shape[0], u.shape[1], 3))
    flow_1[:, :, 0] = u
    flow_1[:, :, 1] = v
    flow_1[:, :, 2] = np.ones((u.shape[0], u.shape[1]))

    if visualize:

        hsv = np.zeros(cv2.imread(img1).shape, dtype=np.uint8)
        hsv[:, :, 0] = 255
        hsv[:, :, 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow('pyflow_output.png', rgb)
        cv2.waitKey(0)

    return flow, flow_1


def farneback(img1, img2, visualize=True):
    image1 = cv2.imread(img1)
    image2 = cv2.imread(img2)
    previous = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    s = time.time()
    flow = cv2.calcOpticalFlowFarneback(previous, next, flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                                        poly_n=5, poly_sigma=1.2, flags=0)
    e = time.time()
    print('Time Taken: %.2f seconds' % (e - s))

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    flow_1 = np.ndarray((u.shape[0], u.shape[1], 3))
    flow_1[:, :, 0] = u
    flow_1[:, :, 1] = v
    flow_1[:, :, 2] = np.ones((u.shape[0], u.shape[1]))


    if visualize:
        hsv = np.zeros_like(image1)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow('farneback_output', bgr)
        cv2.waitKey(0)

    return flow, flow_1


def horn_schunck(img1, img2, visualize = True):
    image1 = cv2.imread(img1)
    im1 = imageio.imread(img1, as_gray=True)
    im2 = imageio.imread(img2, as_gray=True)

    s = time.time()
    U, V = HornSchunck(im1, im2, alpha=1.0, Niter=100)
    e = time.time()
    print('Time Taken: %.2f seconds' % (e - s))

    flow = np.concatenate((U[..., None], V[..., None]), axis=2)

    flow_1 = np.ndarray((U.shape[0], U.shape[1], 3))
    flow_1[:, :, 0] = U
    flow_1[:, :, 1] = V
    flow_1[:, :, 2] = np.ones((U.shape[0], U.shape[1]))

    if visualize:
        #compareGraphs(U, V, im2)
        hsv = np.zeros_like(image1)
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow('horn_schunck_output', bgr)
        cv2.waitKey(0)

    return flow, flow_1


def Lucas_kanade(img1, img2, visualize=True):

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    # Take first frame and find corners in it
    old_frame = cv2.imread(img1)

    frame = cv2.imread(img2)

    flow = np.zeros((old_frame.shape[0], old_frame.shape[1], 2))

    flow_1 = np.zeros((old_frame.shape[0], old_frame.shape[1], 3))

    u = np.zeros([old_frame.shape[0], old_frame.shape[1]])
    v = np.zeros([old_frame.shape[0], old_frame.shape[1]])

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    s = time.time()
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    e = time.time()

    print('Time Taken: %.2f seconds' % (e - s))
    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    dpi = 80
    im = np.array(Image.open(img1))
    height = im.shape[1]
    width = im.shape[0]
    fig = plt.figure(figsize=(height / dpi, width / dpi), dpi=dpi)

    colors = 'rgb'
    c = colors[2]

    for idx, good_point in enumerate(good_old):
        old_gray_x = good_point[1]
        old_gray_y = good_point[0]
        frame_gray_x = good_new[idx][1]
        frame_gray_y = good_new[idx][0]

        flow_1[int(old_gray_x), int(old_gray_y)] = np.array([frame_gray_x - old_gray_x, frame_gray_y - old_gray_y, 1])
        u[int(old_gray_x), int(old_gray_y)] = np.array([(frame_gray_x - old_gray_x)])
        v[int(old_gray_x), int(old_gray_y)] = np.array([(frame_gray_y - old_gray_y)])
        plt.arrow(old_gray_y, old_gray_x, int(frame_gray_y)*0.1, int(frame_gray_x)*0.1, head_width=5, head_length=5, color=c)

    if visualize:
        plt.imshow(im,cmap='gray')
        plt.axis('off')
        plt.show()

    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    flow_1 = np.ndarray((u.shape[0], u.shape[1], 3))
    flow_1[:, :, 0] = u
    flow_1[:, :, 1] = v
    flow_1[:, :, 2] = np.ones((u.shape[0], u.shape[1]))

    return flow, flow_1


img1 = '../datasets/kitti/images/000045_10.png'
img2 = '../datasets/kitti/images/000045_11.png'

# flow, flow1 = Coarse2Fine(img1, img2, visualize=True)
# flow, flow1 = farneback(img1,img2, visualize=True)
# flow, flow1 = horn_schunck(img1,img2)
# flow, flow1 = Lucas_kanade(img1, img2)

