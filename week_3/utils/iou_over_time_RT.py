# matplotlib.use("TkAgg")
from metrics.evaluation_funcs import *
from PIL import Image
from matplotlib import patches
from matplotlib import pyplot as plt
import matplotlib
import cv2
from matplotlib.animation import FuncAnimation
import os
from tracking_utils import *
from colors import *


def iou_over_time_RT_yolo(gt_dir, detections, begin, end):
    gt_video = Video(Video().getgroundTruthown(gt_dir, begin, end))

    yolo_video = Video(Video().getgroundTruthown(detections, begin, end))

    iou_by_frame_yolo = iou_overtime(gt_video, yolo_video, thres=0.5)

    num_framesyolo = len(iou_by_frame_yolo)

    x = []
    # fig = plt.figure()
    ax = plt.subplot(212)
    Ln, = ax.plot(iou_by_frame_yolo)
    ax.set_xlim([0, (end - begin)])
    ax.set_xlabel('frames')
    ax.set_ylabel('IOU')
    ax1 = plt.subplot(211)
    plt.ion()

    for i in range(num_framesyolo):
        x.append(iou_by_frame_yolo[i])
        Ln.set_ydata(x)
        Ln.set_xdata(range(len(x)))

        detections_bboxes = yolo_video.get_frame_by_id(begin + i)

        gt_bboxes = gt_video.get_frame_by_id(begin + i)
        path = '../datasets/train/S03/c010/frames/image' + str(begin + i) + '.jpg'

        im = np.array(Image.open(path), dtype=np.uint8)
        ax1.imshow(im)

        iouframe, TP, FP, FN = iou_frame(detections_bboxes, gt_bboxes, thres=0.5)

        for bbox in gt_bboxes.bboxes:
            ground = patches.Rectangle(bbox.top_left,
                                       bbox.width, bbox.height,
                                       linewidth=1.75, edgecolor='g', facecolor='none', label='groundtruth')
            ax1.add_patch(ground)

        for bbox_noisy in detections_bboxes.bboxes:
            bb = bbox_noisy.to_result()
            detec = patches.Rectangle(bbox_noisy.top_left,
                                      bbox_noisy.width, bbox_noisy.height,
                                      linewidth=1.5, edgecolor='r', facecolor='none', label='detections')

            ax1.add_patch(detec)

            plt.text(bbox_noisy.top_left[0], bbox_noisy.top_left[1], s='car', color='white',
                     verticalalignment='top', bbox={'color': 'red', 'pad': 0})

        plt.legend(handles=[ground, detec], loc="lower left", prop={'size': 6})
        ax1.axis('off')
        plt.title('IOU over time YOLO v3', fontsize=18)
        plt.show()
        plt.pause(0.01)
        ax1.clear()


def iou_over_time_RT_retinanet(gt_dir, detections, begin, end):
    gt_video = Video(Video().getgroundTruthown(gt_dir, begin, end))

    yolo_video = Video(Video().getgroundTruthown(detections, begin, end))

    iou_by_frame_yolo = iou_overtime(gt_video, yolo_video, thres=0.5)

    num_framesyolo = len(iou_by_frame_yolo)

    labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
                       7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
                       12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep',
                       19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
                       25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
                       31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
                       36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
                       41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
                       48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
                       54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
                       60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
                       66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
                       72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
                       78: 'hair drier', 79: 'toothbrush'}

    x = []
    # fig = plt.figure()
    ax = plt.subplot(212)
    Ln, = ax.plot(iou_by_frame_yolo)
    ax.set_xlim([0, (end - begin)])
    ax.set_xlabel('frames')
    ax.set_ylabel('IOU')
    ax1 = plt.subplot(211)
    plt.ion()

    for i in range(num_framesyolo):
        x.append(iou_by_frame_yolo[i])
        Ln.set_ydata(x)
        Ln.set_xdata(range(len(x)))

        detections_bboxes = yolo_video.get_frame_by_id(begin + i)

        gt_bboxes = gt_video.get_frame_by_id(begin + i)
        path = '../datasets/train/S03/c010/frames/image' + str(begin + i) + '.jpg'

        im = np.array(Image.open(path), dtype=np.uint8)
        ax1.imshow(im)

        iouframe, TP, FP, FN = iou_frame(detections_bboxes, gt_bboxes, thres=0.5)

        for bbox in gt_bboxes.bboxes:
            ground = patches.Rectangle(bbox.top_left,
                                       bbox.width, bbox.height,
                                       linewidth=1.75, edgecolor='g', facecolor='none', label='groundtruth')
            ax1.add_patch(ground)

        for bbox_noisy in detections_bboxes.bboxes:
            bb = bbox_noisy.to_result()
            detec = patches.Rectangle(bbox_noisy.top_left,
                                      bbox_noisy.width, bbox_noisy.height,
                                      linewidth=1.5, edgecolor='r', facecolor='none', label='detections')

            ax1.add_patch(detec)

            # plt.text(bbox_noisy.top_left[0], bbox_noisy.top_left[1], s='car', color='white',
            #         verticalalignment='top', bbox={'color': 'red', 'pad': 0})
            # caption = "{} {:.3f}".format(labels_to_names[label], score)
        plt.legend(handles=[ground, detec], loc="lower left", prop={'size': 6})
        ax1.axis('off')
        plt.title('IOU over time RetinaNet', fontsize=18)
        plt.show()
        plt.pause(0.01)
        ax1.clear()


def iou_over_time_RT_mask_rcnn(gt_dir, detections, begin, end):

    gt_video = Video(Video().getgroundTruthown(gt_dir, begin, end))

    yolo_video = Video(Video().getgroundTruthown(detections, begin, end))

    iou_by_frame_yolo = iou_overtime(gt_video, yolo_video, thres=0.5)

    num_framesyolo = len(iou_by_frame_yolo)

    x = []
    # fig = plt.figure()
    ax = plt.subplot(212)
    Ln, = ax.plot(iou_by_frame_yolo)
    ax.set_xlim([0, (end - begin)])
    ax.set_xlabel('frames')
    ax.set_ylabel('IOU')
    ax1 = plt.subplot(211)
    plt.ion()

    for i in range(num_framesyolo):
        x.append(iou_by_frame_yolo[i])
        Ln.set_ydata(x)
        Ln.set_xdata(range(len(x)))

        detections_bboxes = yolo_video.get_frame_by_id(begin + i)

        gt_bboxes = gt_video.get_frame_by_id(begin + i)
        path = '../datasets/train/S03/c010/frames/image' + str(begin + i) + '.jpg'

        im = np.array(Image.open(path), dtype=np.uint8)
        ax1.imshow(im)

        iouframe, TP, FP, FN = iou_frame(detections_bboxes, gt_bboxes, thres=0.5)

        for bbox in gt_bboxes.bboxes:
            ground = patches.Rectangle(bbox.top_left,
                                       bbox.width, bbox.height,
                                       linewidth=1.75, edgecolor='g', facecolor='none', label='groundtruth')
            ax1.add_patch(ground)

        for bbox_noisy in detections_bboxes.bboxes:
            bb = bbox_noisy.to_result()
            detec = patches.Rectangle(bbox_noisy.top_left,
                                      bbox_noisy.width, bbox_noisy.height,
                                      linewidth=1.5, edgecolor='r', facecolor='none', label='detections')

            ax1.add_patch(detec)

            # plt.text(bbox_noisy.top_left[0], bbox_noisy.top_left[1], s='car', color='white',
            #         verticalalignment='top', bbox={'color': 'red', 'pad': 0})

        plt.legend(handles=[ground, detec], loc="lower left", prop={'size': 6})
        ax1.axis('off')
        plt.title('IOU over time Mask R-CNN', fontsize=18)
        plt.show()
        plt.pause(0.01)
        ax1.clear()


def detection_visualization(gt_dir, detections, begin, end):
    gt_video = Video(Video().getgroundTruthown(gt_dir, begin, end))

    yolo_video = Video(Video().getgroundTruthown(detections, begin, end))

    iou_by_frame_yolo = iou_overtime(gt_video, yolo_video, thres=0.5)

    num_framesyolo = len(iou_by_frame_yolo)

    # fig = plt.figure()
    ax1 = plt.subplot()
    plt.ion()

    for i in range(num_framesyolo):

        detections_bboxes = yolo_video.get_frame_by_id(begin + i)

        gt_bboxes = gt_video.get_frame_by_id(begin + i)
        path = '../datasets/train/S03/c010/frames/image' + str(begin + i) + '.jpg'

        im = np.array(Image.open(path), dtype=np.uint8)
        ax1.imshow(im)

        iouframe, TP, FP, FN = iou_frame(detections_bboxes, gt_bboxes, thres=0.5)

        for bbox in gt_bboxes.bboxes:
            ground = patches.Rectangle(bbox.top_left,
                                       bbox.width, bbox.height,
                                       linewidth=1.75, edgecolor='g', facecolor='none', label='groundtruth')
            ax1.add_patch(ground)

        for bbox_noisy in detections_bboxes.bboxes:
            bb = bbox_noisy.to_result()
            detec = patches.Rectangle(bbox_noisy.top_left,
                                      bbox_noisy.width, bbox_noisy.height,
                                      linewidth=1.5, edgecolor='r', facecolor='none', label='detections')

            ax1.add_patch(detec)

            # plt.text(bbox_noisy.top_left[0], bbox_noisy.top_left[1], s='car', color='white',
            #         verticalalignment='top', bbox={'color': 'red', 'pad': 0})

        plt.legend(handles=[ground, detec], loc="lower left", prop={'size': 6})
        ax1.axis('off')
        plt.title('Mask R-CNN fine-tune', fontsize=18)
        plt.show()
        plt.pause(0.01)
        ax1.clear()


def visualize_detections(detections_video, gt_video, begin, end, save=True, title='detections_'):

    gt = Video(Video().getgroundTruthown(gt_video, begin, end))

    detections = Video(Video().getgroundTruthown(detections_video, begin, end))

    num_frames = end-begin

    fig= plt.figure()

    image = cv2.imread('../datasets/train/S03/c010/frames/image1.jpg')

    height, width = image.shape[:2]

    image = plt.imshow(np.zeros((height, width)), animated=True)

    subplot = [image]

    def animate(i):

        images = '../datasets/train/S03/c010/frames/image' + str(begin + i) + '.jpg'

        img = cv2.imread(images, cv2.COLOR_BGR2RGB)

        detections_bboxes = detections.get_frame_by_id(int(begin) + i)
        gt_bboxes = gt.get_frame_by_id(begin + i)

        for bbox_noisy in detections_bboxes.bboxes:
            cv2.rectangle(img, (int(bbox_noisy.top_left[0]), int(bbox_noisy.top_left[1])), (int(bbox_noisy.top_left[0] + bbox_noisy.width),int(bbox_noisy.top_left[1] + bbox_noisy.height)), (0, 0, 255), 7)


        for bbox_gt in gt_bboxes.bboxes:
            cv2.rectangle(img, (int(bbox_gt.top_left[0]), int(bbox_gt.top_left[1])), (int(bbox_gt.top_left[0] + bbox_gt.width), int(bbox_gt.top_left[1] + bbox_gt.height)),(0, 255, 0), 3)


        subplot[0].set_data(img[:, :, ::-1])
        return subplot

    plt.title('Detection using '+title)
    plt.axis('off')

    ani = FuncAnimation(fig, animate, num_frames, interval=2, blit=True)

    if save:
        ani.save(os.path.join(title+'.gif'), writer='imagemagick')
    else:
        plt.show()
        plt.pause(0.005)


def visualize_detections_with_iou(detections_video, gt_video, begin, end, save=True, title='detections_'):

    gt = Video(Video().getgroundTruthown(gt_video, begin, end))

    detections = Video(Video().getgroundTruthown(detections_video, begin, end))

    num_frames = end-begin

    fig, ax = plt.subplots(2, 1)

    image = cv2.imread('../datasets/train/S03/c010/frames/image1.jpg')

    height, width = image.shape[:2]

    image = ax[0].imshow(np.zeros((height, width)), animated=True)

    iou_by_frame = iou_overtime(gt, detections, thres=0.5)

    iou, = ax[1].plot([], [])

    subplot = [image, iou]

    x = []

    def animate(i):

        images = '../datasets/train/S03/c010/frames/image' + str(begin + i) + '.jpg'

        img = cv2.imread(images, cv2.COLOR_BGR2RGB)

        detections_bboxes = detections.get_frame_by_id(int(begin) + i)
        gt_bboxes = gt.get_frame_by_id(begin + i)

        for bbox_noisy in detections_bboxes.bboxes:
            cv2.rectangle(img, (int(bbox_noisy.top_left[0]), int(bbox_noisy.top_left[1])), (int(bbox_noisy.top_left[0] + bbox_noisy.width),int(bbox_noisy.top_left[1] + bbox_noisy.height)), (0, 0, 255), 7)
        for bbox_gt in gt_bboxes.bboxes:
            cv2.rectangle(img, (int(bbox_gt.top_left[0]), int(bbox_gt.top_left[1])), (int(bbox_gt.top_left[0] + bbox_gt.width), int(bbox_gt.top_left[1] + bbox_gt.height)),(0, 255, 0), 3)

        subplot[0].set_data(img[:, :, ::-1])
        x.append(iou_by_frame[i])
        subplot[1].set_data(range(len(x)), x)

        return subplot

    ax[0].set_title('Detection using '+title)
    ax[0].axis('off')
    ax[1].set_xlim([0, (end - begin)])
    ax[1].set_title('IoU over time')
    ax[1].set_xlabel('frames')
    ax[1].set_ylabel('IOU')

    ani = FuncAnimation(fig, animate, num_frames, interval=2, blit=True)

    if save:
        ani.save(os.path.join(title+'.gif'), writer='imagemagick')
    else:
        plt.show()
        plt.pause(0.005)


def visualize_tracking(tracking_video, begin, end, save=True, title='tracking_'):

    num_frames = end-begin

    fig= plt.figure(figsize=(30, 28))

    image = cv2.imread('../datasets/train/S03/c010/frames/image'+ str(begin) +'.jpg')

    height, width = image.shape[:2]

    image = plt.imshow(np.zeros((height, width)), animated=True)

    subplot = [image]

    def animate(i):

        images = '../datasets/train/S03/c010/frames/image' + str(begin + i) + '.jpg'
        img = cv2.imread(images)

        data = tracking_video[i]
        for bbox in data:
            color = label_color(bbox[0])
            caption_id = "ID: {}".format(bbox[0])
            cv2.rectangle(img, (int(bbox[1]), int(bbox[2])), (int(bbox[1] + bbox[3]),int(bbox[2] + bbox[4])), color, 7)
            cv2.putText(img, caption_id, (int(bbox[1]), int(bbox[2]-10)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 7)

        subplot[0].set_data(img[:, :, ::-1])
        return subplot

    plt.axis('off')

    ani = FuncAnimation(fig, animate, num_frames, interval=2, blit=True)

    if save:
        ani.save(os.path.join(title+'.gif'), writer='imagemagick')
    else:
        plt.show()
        plt.pause(0.005)


def visualize_tracking_kal(tracking_video, begin, end, save=True, title='tracking_'):

    num_frames = end-begin

    fig= plt.figure(figsize=(30, 28))

    image = cv2.imread('../datasets/train/S03/c010/frames/image'+ str(begin) +'.jpg')

    height, width = image.shape[:2]

    image = plt.imshow(np.zeros((height, width)), animated=True)

    subplot = [image]

    def animate(i):

        images = '../datasets/train/S03/c010/frames/image' + str(begin + i) + '.jpg'
        img = cv2.imread(images)

        data = tracking_video[i]
        for bbox in data:
            color = label_color(bbox[0])
            caption_id = "ID: {}".format(bbox[0])
            cv2.rectangle(img, (int(bbox[1]), int(bbox[2])), (int(bbox[1] + bbox[3]),int(bbox[2] + bbox[4])), color, 7)
            cv2.putText(img, caption_id, (int(bbox[1]), int(bbox[2]-10)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 7)
            length = 10
            if np.abs(bbox[7]) + np.abs(bbox[8]) > 0.5:
                cv2.arrowedLine(img, (int(bbox[5]), int(bbox[6])), (int(bbox[5])+(int(bbox[7] * length)), int(bbox[6])+(int(bbox[8]) * length)), color, 5)
        subplot[0].set_data(img[:, :, ::-1])
        return subplot

    plt.axis('off')

    ani = FuncAnimation(fig, animate, num_frames, interval=2, blit=True)

    if save:
        ani.save(os.path.join(title+'.gif'), writer='imagemagick')
    else:
        plt.show()
        plt.pause(0.005)




