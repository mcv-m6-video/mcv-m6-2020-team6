from matplotlib import patches
from PIL import Image
from week_4.metrics.evaluation_funcs import *
from collections import Counter

from week_4.utils.tracking_utils import *
from week_4.optical_flow_off_the_shelf import *
from week_4.metrics.Optical_flow_metrics import *
from week_4.metrics.Optical_flow_visualization import *
import math
from operator import itemgetter

# import pkg_resources
# pkg_resources.require("cv2==4.1.1")
# import cv2

from week_4.utils.sort import Sort, convert_x_to_bbox
from matplotlib import patches
from PIL import Image
from week_3.metrics.evaluation_funcs import *
from week_3.utils.tracking_utils import *


def get_bbox_flow(flow, bbox):
    x0, y0, x1, y1 = bbox.astype("int")
    region1 = flow[y0:y1, x0:x1, 0]
    region2 = flow[y0:y1, x0:x1, 1]
    flow_x = np.median(region1)
    flow_y = np.median(region2)
    return flow_x, flow_y


def tracking_kalman_of(detec_dir, gt, begin, end, num_color, visualize=True):
    detections_list = Video(Video().getgroundTruthown(detec_dir, begin, end))  # 450

    kalman_tracker = Sort()

    cmap = get_cmap(num_color)

    ax1 = plt.subplot()
    plt.ion()
    video_detections = []

    difference_vx = np.array(1)
    difference_vy = np.array(1)

    for frame in detections_list.list_frames:
        print(frame.frame_id)
        detections = []
        frame_num = frame.frame_id

        path_now = '../datasets/train/S03/c010/frames/image' + str(frame_num) + '.jpg'
        path_before = '../datasets/train/S03/c010/frames/image' + str(frame_num - 1) + '.jpg'

        im = np.array(Image.open(path_now), dtype=np.uint8)

        if frame_num > begin:
            flow, flow_1 = farneback(path_before, path_now, visualize=False)
        else:
            flow = np.zeros((im.shape[0], im.shape[1], 2))

        for box_detec in frame.bboxes:
            bbox = np.array([box_detec.top_left[0], box_detec.top_left[1], box_detec.width + box_detec.top_left[0],
                              box_detec.height + box_detec.top_left[1]])
            flow_x, flow_y = get_bbox_flow(flow, bbox)
            bbox = np.hstack((bbox, np.array([flow_x, flow_y])))
            detections.append(bbox)

        trackers = kalman_tracker.update(np.array(detections))

        frame_detections = []

        for track_state in trackers:
            x, y = track_state[0], track_state[1]
            vx, vy = track_state[-4], track_state[-3]
            id = track_state[-1]
            bbox = convert_x_to_bbox(track_state[:-1])
            bbox = bbox.reshape(bbox.shape[1])

            if bbox[0] < 0 or bbox[0] > 1920:
                bbox[0] = 0
            if bbox[2] > 1800:
                bbox[2] = 100

            track_det = np.concatenate((bbox, [id])).astype(np.uint64)

            frame_detections.append([track_det[4], track_det[0], track_det[1], (track_det[2] - track_det[0]),
                                     (track_det[3] - track_det[1])])

            color = cmap(track_det[4])
            detec = patches.Rectangle((track_det[0], track_det[1]), (track_det[2] - track_det[0]),
                                      (track_det[3] - track_det[1]),
                                      linewidth=1.5, edgecolor=color, facecolor='none')
            ax1.add_patch(detec)

            length = 10
            if np.abs(vx) + np.abs(vy) > 0.5:
                ax1.arrow(x, y, vx * length, vy * length, head_width=10, head_length=10, fc=color, ec=color)

            flow_x, flow_y = get_bbox_flow(flow, bbox)
            if np.abs(flow_x) + np.abs(flow_y) > 0.5:
                ax1.arrow(x, y, flow_x * length, flow_y * length, head_width=10, head_length=10, fc='cyan', ec='cyan')

            difference_vx = np.vstack((difference_vx, np.array([vx - flow_x])))
            difference_vy = np.vstack((difference_vy, np.array([vy - flow_y])))

            caption_id = "ID: {}".format(track_det[4])
            ax1.text(track_det[0], track_det[1] - 10, caption_id, color=color)

            ax1.text(10, 60, "Frame: {}".format(frame_num), color='w', fontsize=11)

        video_detections.append(frame_detections)
        if visualize:
            ax1.axis('off')
            ax1.imshow(im)
            plt.show()
            plt.savefig("kalman_filter_images/image_{}".format(frame_num))
            plt.pause(0.001)
            ax1.clear()


    ax1.clear()
    plt.hist(difference_vx, bins=1000)
    plt.title("Kalman Vx - Optical Flow x")
    plt.show()
    plt.savefig("difference_vx")
    print("mean absolute error vx ", np.mean(np.abs(difference_vx)))
    ax1.clear()

    plt.hist(difference_vy, bins=1000)
    plt.title("Kalman Vy - Optical Flow y")
    plt.show()
    plt.savefig("difference_vy")
    print("mean absolute error vx ", np.mean(np.abs(difference_vy)))

    gt_list = Video(Video().getgroundTruthown(gt, begin, end))

    compute_IDF1(video_detections, gt_list)

    print(compute_IDF1(video_detections, gt_list))

    return video_detections
