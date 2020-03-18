from matplotlib import patches
from PIL import Image
from week_3.metrics.evaluation_funcs import *
from collections import Counter
import cv2
from week_3.utils.tracking_utils import *
from week_3.utils.iou_over_time_RT import *


def tracking_overlap(detec_dir, gt, begin, end, num_color, visualize=True):
    tracker = 0
    detec_x_frame = 0
    detections_frame = []
    trackings = []
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

    for frame in detections.list_frames:

        image_num = frame.frame_id

        path = '../datasets/train/S03/c010/frames/image' + str(image_num) + '.jpg'

        hsv = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2HSV)[:, :, 0]

        if not trackings:

            detec_x_frame = len(frame.bboxes)
            tracker = tracker + detec_x_frame
            ids = list(range(0, tracker))

            detections_frame = frame.bboxes
        else:
            old_detections = []
            new_detections = []
            for detec_box in frame.bboxes:

                ious = []

                tracks = [[frame, car_id, boxes, histo] for (frame, car_id, boxes, histo) in trackings if frame == image_num-1
                          or frame == image_num-2 or frame == image_num-3 or frame == image_num-4 or frame == image_num-5]

                hsv_detec = hsv[np.int(detection.top_left[1]):np.int(detection.top_left[1] + detection.height),
                          np.int(detection.top_left[0]):
                          np.int(detection.top_left[0] + detection.width)]

                histogram = np.histogram(hsv_detec, bins=np.linspace(0, 255, 16), density=True)

                for i in tracks:
                    ious.append([i[1], iou_bbox_2(detec_box, i[2]), detec_box, cv2.compareHist(np_hist_to_cv(histogram),
                                                                                               np_hist_to_cv(i[3]), method=cv2.HISTCMP_INTERSECT)])
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

            hsv_roi = hsv[np.int(detection.top_left[0]): np.int(detection.top_left[0] + detection.width),
                      np.int(detection.top_left[1]):np.int(detection.top_left[1] + detection.height)]

            histo = np.histogram(hsv_roi, bins=np.linspace(0, 255, 16), density=True)

            trackings.append([image_num, id, detection, histo])

            frame_detections.append([id, detection.top_left[0], detection.top_left[1], detection.width, detection.height])

            color = cmap(id)

            box = list([detection.top_left[0], detection.top_left[1], detection.width + detection.top_left[0],
                        detection.height + detection.top_left[1]])

            detec = patches.Rectangle(detection.top_left, detection.width, detection.height, linewidth=1.5, edgecolor=color, facecolor='none')
            ax1.add_patch(detec)

            caption_id = "ID: {}".format(id)
            ax1.text(detection.top_left[0], detection.top_left[1]-10, caption_id, color=color)

            ax1.text(10, 60, "Frame: {}".format(image_num), color='w', fontsize=11)

        video_detections.append(frame_detections)

        if visualize:
            ax1.axis('off')
            ax1.imshow(im)
            plt.show()
            plt.pause(0.001)
            ax1.clear()

    gt_list = Video(Video().getgroundTruthown(gt, begin, end))

    print(compute_IDF1(video_detections, gt_list))

    return video_detections

#detec_dir = 'detections_mask_rcnn_fine_tune_25.txt'
#gt = 'annotation_only_cars.txt'
#video_detections = tracking_overlap(detec_dir, gt, 450, 900, 50, visualize=False)
#visualize_tracking(video_detections, 450, 900, save=True, title='tracking_')
