import os
import pickle
import re
import glob
import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm
from preprocessing import *
matplotlib.use('TkAgg')

# Single Gaussian Method
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


def get_gaussian_model(train_list, visualize=False,adaptive=False,rho=0.01,colorspace="gray"):

    if colorspace=="gray":
        color=cv2.COLOR_BGR2GRAY
    if colorspace=="yuv":
        color=cv2.COLOR_BGR2YUV
    if colorspace=="hsv":
        color=cv2.COLOR_BGR2HSV

    num_train = len(train_list)
    ini_frame = cv2.cvtColor(cv2.imread(train_list[0]), color)

    if colorspace == "gray":
        image_list_bg = np.zeros((num_train, ini_frame.shape[0], ini_frame.shape[1]))
    else:
        image_list_bg = np.zeros((num_train, ini_frame.shape[0], ini_frame.shape[1], ini_frame.shape[2]))

    filename = 'mean_std_train.pkl'


    if os.path.exists(filename):
        print()
        with open(filename, 'rb') as file:
            mean, std = pickle.load(file)
    else:

        bar1 = tqdm(total=num_train)

        for i in range(0, num_train):
            image = cv2.cvtColor(cv2.imread(train_list[i]), color)

            if colorspace == "gray":
                image_list_bg[i, :, :] = image
            else:
                image_list_bg[i,:, :, :] = image


            bar1.update(1)

        bar1.close()

        mean = image_list_bg.mean(axis=0)
        std = image_list_bg.std(axis=0)

        with open(filename, 'wb') as f:
            pickle.dump([mean, std], f)

    if adaptive:
        filename_a="mean_std_train_ad.pkl"

        if os.path.exists(filename_a):
            print()
            with open(filename_a, 'rb') as file:
                mean, std = pickle.load(file)

        else:
            mean = (1 - rho) * mean + rho * image_list_bg
            std = np.sqrt(rho * (image_list_bg - mean) ** 2 + (1 - rho) * std ** 2)

            with open(filename_a, 'wb') as f:
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
    foreground = abs(img - mean_model) >= alpha * (std_model + 2)
    return foreground


def fg_mask_single_gaussian_color(frames_path, roi_path, alpha,adaptive=False,rho=0.01,colorspace="gray"):
    [train_list, test_list] = get_frames(frames_path, 0.25)

    if colorspace=="gray":
        color=cv2.IMREAD_GRAYSCALE
    if colorspace=="yuv":
        color=cv2.COLOR_BGR2YUV
    if colorspace=="hsv":
        color=cv2.COLOR_BGR2HSV

    num_train = len(train_list)
    num_test = len(test_list)
    num_frames = num_test + num_train

    ini_frame = cv2.cvtColor(cv2.imread(train_list[0]), color)

    mean, std = get_gaussian_model(train_list, False,adaptive,rho,colorspace)

    filename = 'image_list_fg.pkl'

    if os.path.exists(filename):
        print()
        with open(filename, 'rb') as file:
            image_list_fg = pickle.load(file)
    else:

        bar2 = tqdm(total=num_test)
        if colorspace=="gray":
            image_list_fg = np.zeros((num_test, ini_frame.shape[0], ini_frame.shape[1]))
        else:
            image_list_fg=np.zeros((num_test, ini_frame.shape[0], ini_frame.shape[1],ini_frame.shape[2]))

        roi = cv2.cvtColor(cv2.imread(roi_path), color)

        for j in range(0, num_test):
            image = cv2.imread(test_list[j], color)
            if colorspace == "gray":
                image_list_fg[j, :, :] = morphology_operations(roi * frame_mask_single_gaussian(image, mean, std, alpha),
                    kernel_open=(5, 5), kernel_close=(7, 7))
            else:
                image_list_fg[j, :, :,:] = morphology_operations(roi * frame_mask_single_gaussian(image, mean, std, alpha),
                                                           kernel_open=(5, 5), kernel_close=(7, 7))

            bar2.update(1)
        bar2.close()

        #with open(filename, 'wb') as f:
            #pickle.dump(image_list_fg, f)


    return image_list_fg

def visualize_mask(detections_video, mask, begin, end, images):

    num_frames = end-begin

    x = []

    fig, (ax1, ax2) = plt.subplots(1, 2)

    fig.suptitle('Foreground detection')

    plt.ion()

    for i in range(num_frames):

        detections_bboxes = detections_video.get_frame_by_id(begin + i)
        ax1.set_title('Detection')
        ax1.imshow(cv2.imread(images[i]))


        for bbox_noisy in detections_bboxes.bboxes:
            bb = bbox_noisy.to_result()
            detec = patches.Rectangle(bbox_noisy.top_left,
                                      bbox_noisy.width, bbox_noisy.height,
                                      linewidth=1.5, edgecolor='r', facecolor='none', label='detections')

            ax1.add_patch(detec)

        ax2.imshow(mask[i])
        ax2.set_title('Mask')

        ax1.axis('off')
        ax2.axis('off')

        plt.show()
        plt.pause(0.005)
        ax1.clear()
        ax2.clear()

def background_gaussian():

    #Background Substraction Non-adaptive
    #Display of 25 frames as example.

    frames_dir = '../datasets/train/S03/c010/frames/'
    roi_path = '../datasets/train/S03/c010/roi.jpg'
    gt_dir = 'annotation.txt'

    train_list, test_list = get_frames(frames_dir, trainbackground=0.25)
    image_list_fg = fg_mask_single_gaussian(frames_dir, roi_path, alpha=4,adaptive=False, colorspace="hsv")

    #listbboxes, listofmask, video_fg = connected_component_test(image_list_fg[0:25], min_area=1500)

    #gt_video = Video(Video().getgroundTruthown(gt_dir, 0, 2141))

    #visualize_mask(video_fg, listofmask, 0, 25, test_list[0:25])

    #mAP(gt_video, video_fg, True, fname='precision_recall.png')

def background_adaptive_gaussian():
    frames_dir = '../datasets/train/S03/c010/frames/'
    roi_path = '../datasets/train/S03/c010/roi.jpg'
    gt_dir = 'annotation.txt'

    train_list, test_list = get_frames(frames_dir, trainbackground=0.25)
    image_list_fg = fg_mask_single_gaussian(frames_dir, roi_path, alpha=4, adaptive=True,rho=0.01)

    listbboxes, listofmask, video_fg = connected_component_test(image_list_fg, min_area=1500)

    gt_video = Video(Video().getgroundTruthown(gt_dir, 0, 2141))

    visualize_mask(video_fg, listofmask, 0, 25, test_list[0:25])

    #mAP(gt_video, video_fg, True, fname='precision_recall.png')

"""
frames_dir = '../../datasets/train/S03/c010/frames/'
roi_path = '../../datasets/train/S03/c010/roi.jpg'
prova = fg_mask_single_gaussian(frames_dir, roi_path)
"""