import glob
import os
import pickle
import re

import cv2
import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt, patches
from metrics.evaluation_funcs import iou_overtime
from matplotlib.animation import FuncAnimation
import gc
import copy
from utils.preprocessing import connected_component_test, morphology_operations


def get_frames(frames_path, trainbackground=0.25):
    filelist = os.listdir(frames_path)
    filelist.sort(key=lambda f: int(re.sub('\D', '', f)))

    img_list = []

    for frame in filelist:
        img = glob.glob(frames_path + frame)
        img_list.append(img[0])
    # 25% training background
    num_frames = len(img_list)

    num_train = round(trainbackground * num_frames)
    num_test = round(num_frames - num_train)

    train_list = img_list[:num_train]
    test_list = img_list[num_train:]

    return train_list, test_list


def get_gaussian_model(train_list, visualize=False, colorspace="gray", cam='c010'):
    if colorspace == "gray":
        color = cv2.COLOR_BGR2GRAY
        filename = 'mtsc/detections_bg/mean_std_train_' + cam + '.pkl'

    if colorspace == "yuv":
        color = cv2.COLOR_BGR2YUV
        filename = 'mean_std_train_yuv.pkl'

    if colorspace == "hsv":
        color = cv2.COLOR_BGR2HSV
        filename = 'mean_std_train_hsv.pkl'
    gc.collect()
    num_train = len(train_list)
    ini_frame = cv2.cvtColor(cv2.imread(train_list[0]), color)

    if colorspace == "gray":
        image_list_bg_mean = np.zeros((ini_frame.shape[0], ini_frame.shape[1]))
        image_list_bg_var = np.zeros((ini_frame.shape[0], ini_frame.shape[1]))
    else:
        image_list_bg_mean = np.zeros((ini_frame.shape[0], ini_frame.shape[1], ini_frame.shape[2]))
        image_list_bg_var = np.zeros((ini_frame.shape[0], ini_frame.shape[1], ini_frame.shape[2]))

    if os.path.exists(filename):
        print()
        with open(filename, 'rb') as file:
            mean, std = pickle.load(file)
    else:
        bar1 = tqdm(total=num_train)

        for i in range(0, num_train):
            image = cv2.cvtColor(cv2.imread(train_list[i]), color)
            image_list_bg_mean = image_list_bg_mean + image
            bar1.update(1)

        bar1.close()

        mean = image_list_bg_mean / num_train

        bar1 = tqdm(total=num_train)

        for i in range(0, num_train):
            image = cv2.cvtColor(cv2.imread(train_list[i]), color)
            image_list_bg_var = image_list_bg_var + (image - mean) ** 2
            bar1.update(1)

        std = image_list_bg_var / num_train

        bar1.close()
        with open(filename, 'wb') as f:
            pickle.dump([mean, std], f)

    if visualize:
        plt.imshow(mean, cmap="gray")
        plt.axis('off')
        plt.show()
        plt.imshow(std, cmap="gray")
        plt.axis('off')
        plt.show()

    return mean, std


def frame_mask_single_gaussian(img, mean_model, std_model, alpha):
    foreground = abs(img - mean_model) >= alpha * (np.sqrt(std_model) + 2)
    return foreground


def fg_mask_single_gaussian(frames_path, roi_path, alpha, adaptive=False, rho=0.01, colorspace="gray", cam='c010'):
    [train_list, test_list] = get_frames(frames_path, 0.25)

    if colorspace == "gray":
        color = cv2.COLOR_BGR2GRAY
    if colorspace == "yuv":
        color = cv2.COLOR_BGR2YUV
    if colorspace == "hsv":
        color = cv2.COLOR_BGR2HSV
    gc.collect()
    num_train = len(train_list)
    num_test = len(test_list)
    num_frames = num_test + num_train

    ini_frame = cv2.cvtColor(cv2.imread(train_list[0]), color)

    mean, std = get_gaussian_model(train_list, False, colorspace, cam)

    filename = 'image_list_fg.pkl'
    image_list_fg = []
    gc.collect()
    if os.path.exists(filename):
        print()
        with open(filename, 'rb') as file:
            image_list_fg = pickle.load(file)
    else:

        bar2 = tqdm(total=num_test)

        if colorspace == "gray":
            image_list_fg = np.zeros((num_test, ini_frame.shape[0], ini_frame.shape[1]))
        else:
            image_list_fg = np.zeros((num_test, ini_frame.shape[0], ini_frame.shape[1], ini_frame.shape[2]))

        roi = cv2.cvtColor(cv2.imread(roi_path), color)

        for j in range(0, num_test):
            image = cv2.cvtColor(cv2.imread(test_list[j]), color)

            image_list_fg[j, :, :] = morphology_operations(roi * frame_mask_single_gaussian(image, mean, std, alpha),
                                                           kernel_open=(5, 5), kernel_close=(50, 50))
            if adaptive:
                mean = (1 - rho) * mean + rho * image
                std = np.sqrt(rho * (image - mean) ** 2 + (1 - rho) * std ** 2)

            bar2.update(1)
        bar2.close()

    return image_list_fg


def background_adaptive_gaussian(frames_dir, roi_path, gt_dir, filename, cam='c010', num_frame=536):
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            video_fg = pickle.load(file)
    else:
        image_list_fg = fg_mask_single_gaussian(frames_dir, roi_path, alpha=4.75, adaptive=True, rho=1, cam=cam)

        video_fg = connected_component_test(image_list_fg, min_area=1500, num_frame=num_frame)
        listbboxes = []
        image_list_fg = []
        listofmask = []
        gc.collect()
        with open(filename, 'wb') as f:
            pickle.dump(video_fg, f)

    # visualize_mask(video_fg, listofmask, 536, 636, test_list[0:100])

    # visualize_mask(video_fg, listofmask, 536, 636, test_list[0:100], True, title='Task_21_bs_adaptive')

    return video_fg


def visualize_mask(gt_video, begin, end, images, method):
    num_frames = end - begin

    x = []
    fig = plt.figure()
    ax1 = plt.subplot()
    plt.ion()

    for i in range(num_frames):

        gt_bboxes = gt_video.get_frame_by_id(begin + i)

        ax1.imshow(cv2.imread(images[i]))

        for bbox in gt_bboxes.bboxes:
            ground = patches.Rectangle(bbox.top_left,
                                       bbox.width, bbox.height,
                                       linewidth=1.75, edgecolor='g', facecolor='none', label='groundtruth')
            ax1.add_patch(ground)

        """for bbox_noisy in detections_bboxes.bboxes:
            bb = bbox_noisy.to_result()
            detec = patches.Rectangle(bbox_noisy.top_left,
                                      bbox_noisy.width, bbox_noisy.height,
                                      linewidth=1.5, edgecolor='r', facecolor='none', label='detections')

            ax1.add_patch(detec)"""

        ax1.axis('off')
        plt.title('Background subtraction using ' + method)
        plt.show()
        plt.pause(0.005)
        ax1.clear()


def visualize_mask2(detect, gt_video, begin, end, images, method):
    num_frames = end - begin

    x = []
    fig = plt.figure()
    ax1 = plt.subplot()
    plt.ion()

    for i in range(num_frames):

        img = cv2.imread(images[i])
        gt_bboxes = gt_video.get_frame_by_id(begin + i)
        dt_bboxes = detect.get_frame_by_id(begin + i)

        # for bbox_noisy in dt_bboxes.bboxes:
        # cv2.rectangle(img, (bbox_noisy.top_left[0],bbox_noisy.top_left[1]),(bbox_noisy.top_left[0]+bbox_noisy.width,bbox_noisy.top_left[1]+bbox_noisy.height), (255, 0, 0), 5)
        for bbox_gt in gt_bboxes.bboxes:
            cv2.rectangle(img, (int(bbox_gt.top_left[0]), int(bbox_gt.top_left[1])),
                          (int(bbox_gt.top_left[0] + bbox_gt.width), int(bbox_gt.top_left[1] + bbox_gt.height)),
                          (0, 255, 0), 10)

        ax1.imshow(img)

        ax1.axis('off')
        plt.title('Background subtraction using ' + method)
        plt.show()
        plt.pause(0.005)
        ax1.clear()


def background_subtraction_sota(frames_dir, roi_path, filename, ini, method):
    if method == 'MOG':
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=100, nmixtures=2)

    elif method == 'MOG2':
        fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=15)

    elif method == 'KNN':
        fgbg = cv2.createBackgroundSubtractorKNN(100, 550)

    train_list, test_list = get_frames(frames_dir, trainbackground=0.25)

    num_test = len(test_list)
    ini_frame = cv2.imread(train_list[0])

    mask = np.zeros((num_test, ini_frame.shape[0], ini_frame.shape[1]))
    roi = cv2.cvtColor(cv2.imread(roi_path), cv2.COLOR_BGR2GRAY)

    images_morpho = []

    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            video_fg = pickle.load(file)
    else:

        for j in range(0, num_test - 1):

            frame = cv2.imread(test_list[j], cv2.IMREAD_GRAYSCALE)
            if method == 'MOG2':
                fgmask = fgbg.apply(frame, 0.01)
            else:
                fgmask = fgbg.apply(frame)

            mask_morpho = morphology_operations(roi * fgmask, kernel_open=(5, 5), kernel_close=(50, 50))

            mask[j, :, :] = copy.deepcopy(mask_morpho)

            images_morpho.append(mask_morpho)

        video_fg = connected_component_test(mask, min_area=1500, num_frame=ini)

        with open(filename, 'wb') as f:
            pickle.dump(video_fg, f)

    return video_fg


def visualize_mask(detections_video, mask, begin, end, images, save=True, title='Task1_bs'):
    num_frames = end - begin

    fig, ax = plt.subplots(1, 2)

    fig.suptitle('Foreground detection')

    image = cv2.imread(images[0], 0)
    height, width = image.shape[:2]

    image = ax[0].imshow(np.zeros((height, width)), animated=True)
    mask1 = ax[1].imshow(mask[0], animated=True)

    subplot = [image, mask1]

    def animate(i):
        img = cv2.imread(images[i])
        bs = mask[i]
        detections_bboxes = detections_video.get_frame_by_id(begin + i)
        for bbox_noisy in detections_bboxes.bboxes:
            cv2.rectangle(img, (bbox_noisy.top_left[0], bbox_noisy.top_left[1]), (bbox_noisy.top_left[0]
                                                                                  + bbox_noisy.width,
                                                                                  bbox_noisy.top_left[
                                                                                      1] + bbox_noisy.height),
                          (255, 0, 0), 10)
        subplot[0].set_data(img)
        subplot[1].set_data(bs)
        return subplot

    ax[0].set_title('Detection')
    ax[1].set_title('Mask')
    ax[0].axis('off')
    ax[1].axis('off')

    ani = FuncAnimation(fig, animate, num_frames, interval=2, blit=True)

    if save:
        ani.save(os.path.join(title + '.gif'), writer='imagemagick')
    else:
        plt.show()
        plt.pause(0.005)
        ax[0].clear()
        ax[1].clear()


def visualize_mask_alpha(detections_video, gt_video, mask, begin, end, alpha, images, save=True, title='Task12_bs'):
    num_frames = end - begin

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    f_ax0 = fig.add_subplot(gs[0, 0])
    f_ax1 = fig.add_subplot(gs[0, 1])
    f_ax2 = fig.add_subplot(gs[1, :])

    fig.suptitle('Background subtraction using alpha ' + alpha)

    x = []

    image = cv2.imread(images[0], 0)
    height, width = image.shape[:2]

    image = f_ax0.imshow(np.zeros((height, width)), animated=True)

    iou_by_frame = iou_overtime(gt_video, detections_video, thres=0.5)

    mask1 = f_ax1.imshow(mask[0], cmap='gray')

    iou, = f_ax2.plot([], [])

    subplot = [image, mask1, iou]

    def animate(i):
        img = cv2.imread(images[i])
        bs = mask[i]
        detections_bboxes = detections_video.get_frame_by_id(begin + i)
        gt_bboxes = gt_video.get_frame_by_id(begin + i)

        for bbox_noisy in detections_bboxes.bboxes:
            cv2.rectangle(img, (bbox_noisy.top_left[0], bbox_noisy.top_left[1]),
                          (bbox_noisy.top_left[0] + bbox_noisy.width, bbox_noisy.top_left[1] + bbox_noisy.height),
                          (255, 0, 0), 10)
        for bbox_gt in gt_bboxes.bboxes:
            cv2.rectangle(img, (int(bbox_gt.top_left[0]), int(bbox_gt.top_left[1])),
                          (int(bbox_gt.top_left[0] + bbox_gt.width), int(bbox_gt.top_left[1] + bbox_gt.height)),
                          (0, 255, 0), 10)

        #   cv2.putText(bs, alpha, ((bbox_gt.top_left[0] + bbox_gt.width)-20, (bbox_gt.top_left[1] + bbox_gt.height)-20), cv2.FONT_HERSHEY_SIMPLEX, 5, (255,0,255), 2)
        x.append(iou_by_frame[i])
        subplot[0].set_data(img)
        subplot[1].set_data(bs)
        subplot[2].set_data(range(len(x)), x)

        return subplot

    f_ax0.set_title('Detection')
    f_ax1.set_title('Mask')
    f_ax2.set_xlim([0, (end - begin)])
    f_ax2.set_title('IoU over time')
    f_ax2.set_xlabel('frames')
    f_ax2.set_ylabel('IOU')
    f_ax0.axis('off')
    f_ax1.axis('off')

    ani = FuncAnimation(fig, animate, num_frames, interval=2, blit=True)

    if save:
        ani.save(os.path.join(title + '.gif'), writer='imagemagick')
    else:
        plt.show()
        plt.pause(0.005)
        f_ax0.clear()
        f_ax1.clear()
        f_ax2.clear()


def visualize_mask_alpha_rho(detections_video, gt_video, mask, begin, end, alpha, rho, images, save=True,
                             title='Task21_adaptive'):
    num_frames = end - begin

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(1, 2)

    f_ax0 = fig.add_subplot(gs[0, 0])
    f_ax1 = fig.add_subplot(gs[0, 1])

    fig.suptitle('Background subtraction using alpha ' + alpha + ' and rho ' + rho)

    image = cv2.imread(images[0], 0)
    height, width = image.shape[:2]

    image = f_ax0.imshow(np.zeros((height, width)), animated=True)

    mask1 = f_ax1.imshow(mask[0], cmap='gray')

    subplot = [image, mask1]

    def animate(i):
        img = cv2.imread(images[i])
        bs = mask[i]
        detections_bboxes = detections_video.get_frame_by_id(begin + i)
        gt_bboxes = gt_video.get_frame_by_id(begin + i)

        for bbox_noisy in detections_bboxes.bboxes:
            cv2.rectangle(img, (bbox_noisy.top_left[0], bbox_noisy.top_left[1]),
                          (bbox_noisy.top_left[0] + bbox_noisy.width, bbox_noisy.top_left[1] + bbox_noisy.height),
                          (255, 0, 0), 10)
        for bbox_gt in gt_bboxes.bboxes:
            cv2.rectangle(img, (int(bbox_gt.top_left[0]), int(bbox_gt.top_left[1])),
                          (int(bbox_gt.top_left[0] + bbox_gt.width), int(bbox_gt.top_left[1] + bbox_gt.height)),
                          (0, 255, 0), 10)

        subplot[0].set_data(img)
        subplot[1].set_data(bs)

        return subplot

    f_ax0.set_title('Detection')
    f_ax1.set_title('Mask')
    f_ax0.axis('off')
    f_ax1.axis('off')

    ani = FuncAnimation(fig, animate, num_frames, interval=2, blit=True)

    if save:
        ani.save(os.path.join(title + '.gif'), writer='imagemagick')
    else:
        plt.show()
        plt.pause(0.005)
        f_ax0.clear()
        f_ax1.clear()
