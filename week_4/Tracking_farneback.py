
from matplotlib import patches
from PIL import Image
from week_4.metrics.evaluation_funcs import *
from collections import Counter
import cv2
from week_4.utils.tracking_utils import *
from week_4.optical_flow_off_the_shelf import *
from week_4.metrics.Optical_flow_metrics import *
from week_4.metrics.Optical_flow_visualization import *


def tuple_comparison(tuple1, tuple2):

    if tuple1 == tuple2:
        result = 1
    else:
        result = 0
    return result


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


        if image_num >= begin + 1:
            path1 = '../datasets/train/S03/c010/frames/image' + str(image_num - 1) + '.jpg'

            flow, flow1 = optical_flow_estimation(path, path1, visualize=False)
            #OF_quiver_visualization_plot(path1, flow1, step=2)

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
                    x = round(detection.top_left[0])
                    width = round(detection.width + x)
                    y = round(detection.top_left[1])
                    height = round(detection.height + y)

                    detec_flow = flow[x: width, y:height]

                    average_x = np.mean(np.median(detec_flow[:, :, 0]))
                    average_y = np.mean(np.median(detec_flow[:, :, 1]))

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
                        OF_2 = [0, 0]
                        OF_final = [OF[0] + OF_2[0], OF[1] + OF_2[1]]
                    else:
                        OF = [x_dir, y_dir]
                        OF_2 = [x_dir1, y_dir1]
                        OF_final = [OF[0] + OF_2[0], OF[1] + OF_2[1]]
                else:
                    OF = [0, 0]
                    OF_2 = [0, 0]
                    OF_final = [OF[0] + OF_2[0], OF[1] + OF_2[1]]

                ious = []

                tracks = [[frame, car_id, boxes, OF_final] for (frame, car_id, boxes, OF_final) in trackings if frame == image_num-1
                          or frame == image_num-2 or frame == image_num-3]

                for i in tracks:
                    ious.append([i[1], iou_bbox_2(detec_box, i[2]), detec_box, 0])
                    if image_num >= begin + 1:
                        ious.append([i[1], iou_bbox_2(detec_box, i[2]), detec_box, tuple_comparison(i[3], OF_final)])
                if ious:
                    ioubox = max(ious, key=lambda x: (x[1], x[3]))

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

                detec_flow = flow[x: width, y:height,:]

                average_x = np.median(np.median(detec_flow[:, :, 0]))
                average_y = np.median(np.median(detec_flow[:, :, 1]))

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
                    OF_2 = [0, 0]
                    OF_final = [OF[0] + OF_2[0], OF[1] + OF_2[1]]
                else:
                    OF = [x_dir, y_dir]
                    OF_2 = [x_dir1, y_dir1]
                    OF_final = [OF[0] + OF_2[0], OF[1] + OF_2[1]]
            else:
                OF = [0, 0]
                OF_2 = [0, 0]
                OF_final = [OF[0] + OF_2[0], OF[1] + OF_2[1]]

            OF_state.append([id, OF_final])

            hola= len(OF_state)

            values = set(map(lambda x: x[0], OF_state))
            newlist = [[y[1] for y in OF_state if y[0] == x] for x in values]

            import collections

            val_1 = collections.Counter([x for (x, y) in map(tuple, newlist[id][-5:])])
            val_2 = collections.Counter([y for (x, y) in map(tuple, newlist[id][-5:])])

            OF_final = [max(val_1, key=val_1.get), max(val_2, key=val_2.get)]

            trackings.append([image_num, id, detection, OF_final])

            frame_detections.append([id, detection.top_left[0], detection.top_left[1], detection.width, detection.height])

            color = cmap(id)

            box = list([detection.top_left[0], detection.top_left[1], detection.width + detection.top_left[0],
                        detection.height + detection.top_left[1]])

            detec = patches.Rectangle(detection.top_left, detection.width, detection.height, linewidth=1.5, edgecolor=color, facecolor='none')
            ax1.add_patch(detec)

            caption_id = "ID: {}".format(id)
            ax1.text(detection.top_left[0], detection.top_left[1]-10, caption_id, color=color)

            ax1.text(10, 60, "Frame: {}".format(image_num), color='w', fontsize=11)

            if image_num >= begin + 1:
                ax1.quiver(detection.top_left[0]+(detection.width/2), detection.top_left[1]+(detection.height/2), OF_final[0], OF_final[1], scale_units='xy', angles='xy', scale=2, color=color)

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
tracking_overlap(detec_dir, gt, 105, 115, 50, visualize=True)