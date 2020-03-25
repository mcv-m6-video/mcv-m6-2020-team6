
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
#import pkg_resources
#pkg_resources.require("cv2==4.1.1")
#import cv2


def distance(a, b):

    if a == b:
        return 0
    elif (a < 0) and (b < 0) or (a > 0) and (b > 0):
        if a < b:
            return abs(abs(a) - abs(b))
        else:
            return -(abs(abs(a) - abs(b)))
    else:
        return math.copysign((abs(a) + abs(b)),b)


def tuple_comparison(tuple1, tuple2):

    x = distance(tuple1[0], tuple2[0])
    y = distance(tuple1[1], tuple2[1])

    difference = abs(x) + abs(y)

    return difference

"""
def optical_flow_estimation(img1, img2, visualize=True):
    image1 = cv2.imread(img1)
    image2 = cv2.imread(img2)
    previous = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    previous = cv2.GaussianBlur(previous, (5, 5), cv2.BORDER_DEFAULT)
    next = cv2.GaussianBlur(next, (5, 5), cv2.BORDER_DEFAULT)

    s = time.time()
    flow = cv2.calcOpticalFlowFarneback(previous, next, flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                                        poly_n=5, poly_sigma=1.2, flags=0)
    e = time.time()
    flow = cv2.resize(flow, (1920, 1080))
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
"""
def optical_flow_estimation(img1, img2, p0):
    # params for ShiTomasi corner detection


    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    old_frame = cv2.imread(img1)

    frame = cv2.imread(img2)


    flow = np.zeros((old_frame.shape[0], old_frame.shape[1], 2))

    u = np.zeros([old_frame.shape[0], old_frame.shape[1]])
    v = np.zeros([old_frame.shape[0], old_frame.shape[1]])

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    dpi = 80
    im = np.array(Image.open(img1))
    height = im.shape[1]
    width = im.shape[0]
    colors = 'rgb'
    c = colors[2]

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        b, a = new.ravel()
        d, c = old.ravel()
    """
    for idx, good_point in enumerate(good_old):
        old_gray_x = good_point[1]
        old_gray_y = good_point[0]
        frame_gray_x = good_new[idx][1]
        frame_gray_y = good_new[idx][0]

        flow_1[int(old_gray_x), int(old_gray_y)] = np.array([frame_gray_x - old_gray_x, frame_gray_y - old_gray_y, 1])
        u[int(old_gray_x), int(old_gray_y)] = np.array([(frame_gray_x - old_gray_x)])
        v[int(old_gray_x), int(old_gray_y)] = np.array([(frame_gray_y - old_gray_y)])


    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    """
    return (b - d), (a-c)


def tracking_overlap(detec_dir, gt, begin, end, num_color, visualize=True):
    tracker = 0
    detec_x_frame = 0
    detections_frame = []
    trackings = []
    OF_state=[]
    old_detections = []
    new_detections = []
    list_detections = []
    ids = []
    video_detections = []
    fig = plt.figure()
    ax1 = plt.subplot()
    plt.ion()
    cmap = get_cmap(num_color)

    detections = Video(Video().getgroundTruthown(detec_dir, begin, end))  # 450
    tracks = []
    for frame in detections.list_frames:

        image_num = frame.frame_id

        path = '../datasets/train/S03/c010/frames/image' + str(image_num) + '.jpg'
        path1 = '../datasets/train/S03/c010/frames/image' + str(image_num - 1) + '.jpg'
        img = cv2.imread(path)
        mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # flow, flow1 = optical_flow_estimation(path, path1, False)

        if not trackings:

            detec_x_frame = len(frame.bboxes)
            tracker = tracker + detec_x_frame
            ids = list(range(0, tracker))

            detections_frame = frame.bboxes
        else:
            old_detections = []
            new_detections = []
            for detec_box in frame.bboxes:

                if image_num >= begin + 1:

                    direction = []
                    x_dir1 = 0
                    x_dir = 0
                    y_dir1 = 0
                    y_dir = 0
                    x = round(detec_box.top_left[0])
                    width = round(detec_box.width)
                    y = round(detec_box.top_left[1])
                    height = round(detec_box.height)

                    feature_params = dict(maxCorners=1000,
                                          qualityLevel=0.3,
                                          minDistance=7,
                                          blockSize=7)

                    roi= mask[y: y + height, x: x + width]

                    plt.imshow(roi)
                    plt.show()

                    p0 = cv2.goodFeaturesToTrack(roi, **feature_params)


                    average_x, average_y = optical_flow_estimation(path, path1, p0)


                    """
                    detec_flow = flow[ x: width,y:height]

                    average_x = np.mean(np.median(detec_flow[:, :, 0]))

                    average_y = np.mean(np.median(detec_flow[:, :, 1]))
                    """

                    north = 1 if abs(average_x) < abs(average_y) and np.sign(average_y) == -1 else 0.5 if np.sign(
                        average_y) == -1 else 0
                    east = 1 if abs(average_x) > abs(average_y) and np.sign(average_x) == 1 else 0.5 if np.sign(
                        average_x) == 1 else 0
                    west = 1 if abs(average_x) > abs(average_y) and np.sign(average_x) == -1 else 0.5 if np.sign(
                        average_x) == -1 else 0
                    south = 1 if abs(average_x) < abs(average_y) and np.sign(average_y) == 1 else 0.5 if np.sign(
                        average_y) == 1 else 0

                    direction = [north, east, west, south]

                    if direction[0] == 1:
                        x_dir = 0
                        y_dir = -1
                    if direction[1] == 1:
                        x_dir = 1
                        y_dir = 0
                    if direction[2] == 1:
                        x_dir = -1
                        y_dir = 0
                    if direction[3] == 1:
                        x_dir = 0
                        y_dir = 1
                    if direction[0] == 0.5:
                        x_dir1 = 0
                        y_dir1 = -1
                    if direction[1] == 0.5:
                        x_dir1 = 1
                        y_dir1 = 0
                    if direction[2] == 0.5:
                        x_dir1 = -1
                        y_dir1 = 0
                    if direction[3] == 0.5:
                        x_dir1 = 0
                        y_dir1 = 1

                    if np.isnan(average_x).any():
                        OF = [0, 0]
                        OF_final = [OF[0] , OF[1] ]
                    else:
                        OF = [x_dir, y_dir]
                        OF_final = [OF[0] , OF[1] ]
                    
                else:
                    OF = [0, 0]
                    OF_2 = [0, 0]
                    OF_final = [OF[0] + OF_2[0], OF[1] + OF_2[1]]


                ious = []

                tracks = [[frame, car_id, boxes, siam_img] for (frame, car_id, boxes, siam_img) in trackings if
                          frame == image_num - 1 or frame == image_num - 2  or frame == image_num - 3 or frame == image_num - 4
                          or frame == image_num - 5 or frame == image_num - 10]
                for i in tracks:

                    if image_num >= begin + 1:
                        ious.append([i[1], iou_bbox_2(detec_box, i[2]), detec_box, tuple_comparison(i[3], OF_final)])
                    else:
                        ious.append([i[1], iou_bbox_2(detec_box, i[2]), detec_box, 15])


                if ious:


                    #ioubox = max(ious, key=lambda x: (x[1], x[3]))

                    best_candidates = sorted(ious, key=itemgetter(1),reverse=True)[:2]

                    ioubox = min(best_candidates, key=lambda x: (x[3]))
                    #indices = smallest, [index for index, [idee, iou_comp,bbox, direc] in enumerate(best_candidates) if smallest == direc]
                    #ioubox = max(map(best_candidates.__getitem__, indices[1]), key=lambda x:(x[1]))

                if ioubox[1] > 0.5:

                    old_detections.append([image_num, ioubox[0], ioubox[2]])
                else:
                    num_new_id = [ide[1] for ide in trackings]

                    try:
                        num_new_id = max(k for k, v in Counter(num_new_id).items() if v>=5)
                    except (ValueError, RuntimeError, TypeError, NameError):
                        num_new_id = max(k for k, v in Counter(num_new_id).items() if v >= 1)

                    new_detections.append([image_num, num_new_id+1+len(new_detections), detec_box])

            old_detections = sorted(old_detections, key=lambda x: x[1])

            list_detections = [row[2] for row in old_detections]

            list_detections.extend([row[2] for row in new_detections])

            ids = [row[1] for row in old_detections]
            ids.extend(row[1] for row in new_detections)

            detections_frame = list_detections

        frame_detections = []

        im = np.array(Image.open(path), dtype=np.uint8)

        for detection, id in zip(detections_frame, ids):

            if image_num >= begin + 1:
                direction = []
                x = round(detection.top_left[0])
                width = round(detection.width + x)
                y = round(detection.top_left[1])
                height = round(detection.height + y)

                feature_params = dict(maxCorners=100,
                                      qualityLevel=0.3,
                                      minDistance=7,
                                      blockSize=7)

                img = cv2.imread(path)

                mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                roi = mask[y: height, x: width]

                p0 = cv2.goodFeaturesToTrack(roi, **feature_params)
                average_x, average_y = optical_flow_estimation(path, path1, p0)

                """
                detec_flow = flow[ x: width,y:height]

                average_x = np.mean(np.median(detec_flow[:, :, 0]))

                average_y = np.mean(np.median(detec_flow[:, :, 1]))
                """

                north = 1 if abs(average_x) < abs(average_y) and np.sign(average_y) == -1 else 0.5 if np.sign(average_y) == -1 else 0
                east = 1 if abs(average_x) > abs(average_y) and np.sign(average_x) == 1 else 0.5 if np.sign(average_x) == 1 else 0
                west = 1 if abs(average_x) > abs(average_y) and np.sign(average_x) == -1 else 0.5 if np.sign(average_x) == -1 else 0
                south = 1 if abs(average_x) < abs(average_y) and np.sign(average_y) == 1 else 0.5 if np.sign(average_y) == 1 else 0

                direction = [north, east, west, south]

                if direction[0] == 1:
                    x_dir = 0
                    y_dir = -1
                if direction[1] == 1:
                    x_dir = 1
                    y_dir = 0
                if direction[2] == 1:
                    x_dir = -1
                    y_dir = 0
                if direction[3] == 1:
                    x_dir = 0
                    y_dir = 1


                if np.isnan(average_x).any():
                    OF = [0, 0]

                    OF_final = [OF[0] , OF[1] ]
                else:
                    OF = [x_dir, y_dir]
                    OF_2 = [x_dir1, y_dir1]
                    OF_final = [OF[0] , OF[1] ]
            else:
                OF = [0, 0]
                OF_2 = [0, 0]
                OF_final = [OF[0] , OF[1] ]

            OF_state.append([id, OF_final])

            values = set(map(lambda x: x[0], OF_state))
            newlist = [[y[1] for y in OF_state if y[0] == x] for x in values]

            import collections

            val_1 = collections.Counter([x for (x, y) in map(tuple, newlist[id][-10:])])
            val_2 = collections.Counter([y for (x, y) in map(tuple, newlist[id][-10:])])

            OF_final = [max(val_1, key=val_1.get), max(val_2, key=val_2.get)]

            trackings.append([image_num, id, detection, OF_final])

            frame_detections.append([id, detection.top_left[0], detection.top_left[1], detection.width, detection.height])

            color = cmap(id)

            detec = patches.Rectangle(detection.top_left, detection.width, detection.height, linewidth=1.5, edgecolor=color, facecolor='none')
            ax1.add_patch(detec)

            caption_id = "ID: {}".format(id)
            ax1.text(detection.top_left[0], detection.top_left[1]-10, caption_id, color=color)

            ax1.text(10, 60, "Frame: {}".format(image_num), color='w', fontsize=11)

            if image_num >= begin + 1:
                ax1.arrow(detection.top_left[0]+(detection.width/2), detection.top_left[1]+(detection.height/2), OF_final[0]*10, OF_final[1]*10, head_width=10, head_length=10, fc=color, ec=color)

        video_detections.append(frame_detections)

        if visualize:
            ax1.axis('off')
            ax1.imshow(im)
            plt.show()
            plt.pause(0.001)

            ax1.clear()





    gt_list = Video(Video().getgroundTruthown(gt, begin, end))

    compute_IDF1(video_detections, gt_list)


detec_dir = 'detections_mask_rcnn_fine_tune.txt'
gt = 'annotation_only_cars.txt'
tracking_overlap(detec_dir, gt, 1, 2141, 50, visualize=True)