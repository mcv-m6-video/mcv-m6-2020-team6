import os
import pickle
import re
import glob
import cv2
import numpy as np
import matplotlib
from matplotlib import patches
from matplotlib import pyplot as plt
from tqdm import tqdm
from week_2.utils.preprocessing import *
matplotlib.use('TkAgg')
from matplotlib.colors import hsv_to_rgb
from matplotlib.animation import FuncAnimation
import gc


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


def get_gaussian_model(train_list, visualize=False, colorspace="gray"):

    if colorspace=="gray":
        color=cv2.COLOR_BGR2GRAY
        filename = 'mean_std_train.pkl'

    if colorspace=="yuv":
        color=cv2.COLOR_BGR2YUV
        filename = 'mean_std_train_yuv.pkl'

    if colorspace=="hsv":
        color=cv2.COLOR_BGR2HSV
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
            image_list_bg_var =image_list_bg_var+ (image - mean)**2
            bar1.update(1)

        std=image_list_bg_var/num_train

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


def fg_mask_single_gaussian(frames_path, roi_path, alpha,adaptive=False,rho=0.01,colorspace="gray"):
    [train_list, test_list] = get_frames(frames_path, 0.25)

    if colorspace=="gray":
        color=cv2.COLOR_BGR2GRAY
    if colorspace=="yuv":
        color=cv2.COLOR_BGR2YUV
    if colorspace=="hsv":
        color=cv2.COLOR_BGR2HSV
    gc.collect()
    num_train = len(train_list)
    num_test = len(test_list)
    num_frames = num_test + num_train

    ini_frame = cv2.cvtColor(cv2.imread(train_list[0]), color)

    mean, std = get_gaussian_model(train_list, False,colorspace)

    filename = 'image_list_fg.pkl'
    image_list_fg=[]
    gc.collect()
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
            image = cv2.cvtColor(cv2.imread(test_list[j]), color)

            image_list_fg[j, :, :] = morphology_operations(roi * frame_mask_single_gaussian(image, mean, std, alpha),
                                                           kernel_open=(5, 5), kernel_close=(50, 50))
            if adaptive:
                mean = (1 - rho) * mean + rho * image
                std = np.sqrt(rho * (image - mean) ** 2 + (1 - rho) * std ** 2)

            bar2.update(1)
        bar2.close()

    return image_list_fg

def fg_mask_single_gaussian_withload(frames_path, roi_path, alpha,adaptive=False,rho=0.01,colorspace="gray",morpho=True):

    [train_list, test_list] = get_frames(frames_path, 0.25)

    if colorspace=="gray":
        color=cv2.COLOR_BGR2GRAY
    if colorspace=="yuv":
        color=cv2.COLOR_BGR2YUV
    if colorspace=="hsv":
        color=cv2.COLOR_BGR2HSV

    num_train = len(train_list)
    num_test = len(test_list)
    num_frames = num_test + num_train

    ini_frame = cv2.cvtColor(cv2.imread(train_list[0]), color)

    mean, std = get_gaussian_model(train_list, False,colorspace)

    filename = "image_list_fg_alpha" + str(alpha) + "_rho" + str(rho) + "_color" + colorspace + ".obj"

    if os.path.exists(filename):
        #image_list_fg = joblib.load(filename)
        image_list_fg=[]
    else:

        bar2 = tqdm(total=num_test)

        if colorspace=="gray":
            image_list_fg = np.zeros((num_test, ini_frame.shape[0], ini_frame.shape[1]))
        else:
            image_list_fg=np.zeros((num_test, ini_frame.shape[0], ini_frame.shape[1],ini_frame.shape[2]))

        roi = cv2.cvtColor(cv2.imread(roi_path), color)

        for j in range(0, num_test):
            image = cv2.cvtColor(cv2.imread(test_list[j]), color)

            frame_mask=frame_mask_single_gaussian(image, mean, std, alpha)

            if morpho:
                if colorspace!="gray":
                    image_list_fg[j, :, :,:] = morphology_operations(roi * frame_mask, kernel_open=(5, 5),
                                                                     kernel_close=(50, 50))
                else:
                    image_list_fg[j, :, :] = morphology_operations(roi * frame_mask, kernel_open=(5, 5),
                                                                       kernel_close=(50, 50))

            else:
                if colorspace!="gray":
                    image_list_fg[j, :, :,:] = roi * frame_mask

                else:
                    image_list_fg[j, :, :] = roi * frame_mask



            if adaptive:
                mean = (1 - rho) * mean + rho * image
                std = np.sqrt(rho * (image - mean) ** 2 + (1 - rho) * std ** 2)



            bar2.update(1)

        joblib.dump(image_list_fg, filename)
        bar2.close()

    return image_list_fg

def visualize_mask_old(detections_video, mask, begin, end, images):

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

def visualize_mask_2(detections_video, begin, end, images):

    num_frames = end-begin

    x = []

    fig, ax1= plt.subplots(1)

    fig.suptitle('Foreground detection')

    plt.ion()
    bbox_parked=BBox(frame_id=0, top_left=(1452.7, 420.8), width=300.3, height=100.7, confidence=0)
    bbox_parked2=BBox(frame_id=0, top_left=(888, 108), width=309.05, height=15.8, confidence=0)
    bbox_parked3=BBox(frame_id=0, top_left=(518.21, 73), width=80.48, height=90.57, confidence=0)
    for i in range(num_frames):

        detections_bboxes = detections_video.get_frame_by_id(begin + i)
        ax1.set_title('Detection')
        ax1.imshow(cv2.imread(images[i]))

        for bbox_noisy in detections_bboxes.bboxes:
            bb = bbox_noisy.to_result()
            detec = patches.Rectangle(bbox_noisy.top_left,
                                      bbox_noisy.width, bbox_noisy.height,
                                      linewidth=1.5, edgecolor='r', facecolor='none', label='detections')

            detec1 = patches.Rectangle(bbox_parked.top_left,
                                      bbox_parked.width, bbox_parked.height,
                                      linewidth=1.5, edgecolor='y', facecolor='none', label='detections')
            detec2 = patches.Rectangle(bbox_parked2.top_left,
                                       bbox_parked2.width, bbox_parked2.height,
                                       linewidth=1.5, edgecolor='y', facecolor='none', label='detections')
            detec3 = patches.Rectangle(bbox_parked3.top_left,
                                       bbox_parked3.width, bbox_parked3.height,
                                       linewidth=1.5, edgecolor='y', facecolor='none', label='detections')

            ax1.add_patch(detec)
            #ax1.add_patch(detec1)
            #ax1.add_patch(detec2)
            #ax1.add_patch(detec3)



            ax1.add_patch(detec)

        plt.show()
        plt.pause(0.05)
        ax1.clear()

def visualize_mask(detections_video, mask, begin, end, images, save=True, title='Task1_bs'):

    num_frames = end-begin

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
                                                                                  + bbox_noisy.width, bbox_noisy.top_left[1] + bbox_noisy.height), (255, 0, 0), 10)
        subplot[0].set_data(img)
        subplot[1].set_data(bs)
        return subplot

    ax[0].set_title('Detection')
    ax[1].set_title('Mask')
    ax[0].axis('off')
    ax[1].axis('off')

    ani = FuncAnimation(fig, animate, num_frames, interval=2, blit=True)

    if save:
        ani.save(os.path.join(title+'.gif'), writer='imagemagick')
    else:
        plt.show()
        plt.pause(0.005)
        ax[0].clear()
        ax[1].clear()


def visualize_mask_alpha(detections_video, gt_video, mask, begin, end, alpha, images, save=True, title='Task12_bs'):

    num_frames = end-begin

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    f_ax0 = fig.add_subplot(gs[0, 0])
    f_ax1 = fig.add_subplot(gs[0, 1])
    f_ax2 = fig.add_subplot(gs[1, :])

    fig.suptitle('Background subtraction using alpha '+alpha)

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
            cv2.rectangle(img, (bbox_noisy.top_left[0],bbox_noisy.top_left[1]),(bbox_noisy.top_left[0]+bbox_noisy.width,bbox_noisy.top_left[1]+bbox_noisy.height), (255, 0, 0), 10)
        for bbox_gt in gt_bboxes.bboxes:
            cv2.rectangle(img, (int(bbox_gt.top_left[0]), int(bbox_gt.top_left[1])),(int(bbox_gt.top_left[0] + bbox_gt.width),int(bbox_gt.top_left[1]+bbox_gt.height)), (0, 255, 0), 10)

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
        ani.save(os.path.join(title+'.gif'), writer='imagemagick')
    else:
        plt.show()
        plt.pause(0.005)
        f_ax0.clear()
        f_ax1.clear()
        f_ax2.clear()


def visualize_mask_alpha_rho(detections_video, gt_video, mask, begin, end, alpha,rho, images, save=True, title='Task21_adaptive'):

    num_frames = end-begin

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(1, 2)

    f_ax0 = fig.add_subplot(gs[0, 0])
    f_ax1 = fig.add_subplot(gs[0, 1])

    fig.suptitle('Background subtraction using alpha '+alpha+' and rho '+rho)

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
            cv2.rectangle(img, (bbox_noisy.top_left[0],bbox_noisy.top_left[1]),(bbox_noisy.top_left[0]+bbox_noisy.width,bbox_noisy.top_left[1]+bbox_noisy.height), (255, 0, 0), 10)
        for bbox_gt in gt_bboxes.bboxes:
            cv2.rectangle(img, (int(bbox_gt.top_left[0]), int(bbox_gt.top_left[1])),(int(bbox_gt.top_left[0] + bbox_gt.width),int(bbox_gt.top_left[1]+bbox_gt.height)), (0, 255, 0), 10)

        subplot[0].set_data(img)
        subplot[1].set_data(bs)

        return subplot

    f_ax0.set_title('Detection')
    f_ax1.set_title('Mask')
    f_ax0.axis('off')
    f_ax1.axis('off')

    ani = FuncAnimation(fig, animate, num_frames, interval=2, blit=True)

    if save:
        ani.save(os.path.join(title+'.gif'), writer='imagemagick')
    else:
        plt.show()
        plt.pause(0.005)
        f_ax0.clear()
        f_ax1.clear()



def background_gaussian():

    frames_dir = '../datasets/train/S03/c010/frames/'
    roi_path = '../datasets/train/S03/c010/roi.jpg'
    gt_dir = 'annotation_fix.txt'

    alpha=4.75

    train_list, test_list = get_frames(frames_dir, trainbackground=0.25)
    image_list_fg = fg_mask_single_gaussian(frames_dir, roi_path, alpha=alpha,rho=1)

    listbboxes, listofmask, video_fg = connected_component_test(image_list_fg, min_area=1500, num_frame=536)

    gt_video = Video(Video().getgroundTruthown(gt_dir, 536, 2141))

    mAP(gt_video, video_fg, True, fname='precision_recall.png')

    visualize_mask(video_fg, listofmask, 536, 636, test_list[0:100], True, title='Task1_bs')

    #visualize_mask_alpha(video_fg, gt_video, listofmask, 536, 636, str(alpha),test_list[0:100], True, title='Task12_alpha_'+str(alpha))


def background_adaptive_gaussian():

    frames_dir = '../datasets/train/S03/c010/frames/'
    roi_path = '../datasets/train/S03/c010/roi.jpg'
    gt_dir = '../datasets/train/S03/c010/gt/gt.txt'

    train_list, test_list = get_frames(frames_dir, trainbackground=0.25)
    image_list_fg = fg_mask_single_gaussian(frames_dir, roi_path, alpha=4.75, adaptive=True, rho=1)

    listbboxes, listofmask, video_fg = connected_component_test(image_list_fg, min_area=1500, num_frame=536)

    gt_video = Video(Video().getgroundTruthown(gt_dir, 536, 2141))

    #visualize_mask(video_fg, listofmask, 536, 636, test_list[0:100])

    mAP(gt_video, video_fg, True, fname='precision_recall_adaptive.png')

    visualize_mask(video_fg, listofmask, 536, 636, test_list[0:100], True, title='Task_21_bs_adaptive')

    # visualize_mask_alpha_rho(video_fg, gt_video, listofmask, 536, 636, str(alpha),str(rho), test_list[0:100], True, title='Task21_alpha_' + str(alpha)+'_rho_'+str(rho))


def background_gaussian_color(color,adaptive):

    frames_dir = '../datasets/train/S03/c010/frames/'
    roi_path = '../datasets/train/S03/c010/roi.jpg'
    gt_dir = 'annotation_fix.txt'

    if color=="hsv":
        alpha=5.4
        rho=0.11

    else:
        alpha=5.4
        rho=0.01

    train_list, test_list = get_frames(frames_dir, trainbackground=0.25)

    image_list_fg=fg_mask_single_gaussian(frames_dir, roi_path, alpha=alpha,adaptive=adaptive,rho=rho,colorspace=color)
    gt_video = Video(Video().getgroundTruthown(gt_dir, 536, 636))

    listbboxes, listofmask, video_fg = connected_component_test(image_list_fg, min_area=1400,num_frame=536,color=color)
    print("mAP value for min_area:")
    fName="precision_recall_"+color+".png"
    mAP(gt_video, video_fg, True, fname=fName)
    print("\n")
    if adaptive:
        title="Task4_bs_"+color+"_alpha_"+str(alpha)+"_rho_"+str(rho)
        visualize_mask_alpha_rho(video_fg, gt_video, listofmask, 536, 636, str(alpha), str(rho), test_list[0:100], True,
                                 title=title)

    else:
        title="Task4_bs_"+color+"alpha_"+str(alpha)
        visualize_mask_alpha(video_fg, gt_video, listofmask, 536, 636, str(alpha), test_list[0:100], True,
                             title=title)

    #visualize_mask(video_fg, listofmask, 536, 636, test_list[0:100], True, title=title)
