from week_3.utils.sort import Sort, convert_x_to_bbox
from matplotlib import patches
from PIL import Image
from week_3.metrics.evaluation_funcs import *
from week_3.utils.tracking_utils import *


def tracking_kalman(detec_dir, gt, begin, end, num_color,  visualize=True):

    detections_list = Video(Video().getgroundTruthown(detec_dir, begin, end))  # 450

    kalman_tracker = Sort()

    cmap = get_cmap(num_color)

    ax1 = plt.subplot()
    plt.ion()
    video_detections = []

    for frame in detections_list.list_frames:
        print(frame.frame_id)
        detections = []
        frame_num = frame.frame_id
        for box_detec in frame.bboxes:
            boxes = np.array([box_detec.top_left[0], box_detec.top_left[1], box_detec.width + box_detec.top_left[0],
                              box_detec.height + box_detec.top_left[1]])

            detections.append(boxes)

        trackers = kalman_tracker.update(np.array(detections))

        frame_detections = []

        path = '../datasets/train/S03/c010/frames/image' + str(frame_num) + '.jpg'
        path = "/home/marc/M6/M6_VA/week_3/aic19-track1-mtmc-train/train/S01/c001/frames/image_" + str(frame_num).zfill(3) + ".png"

        im = np.array(Image.open(path), dtype=np.uint8)

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

            frame_detections.append([track_det[4], track_det[0], track_det[1], (track_det[2]-track_det[0]),
                                     (track_det[3] - track_det[1])])

            color = cmap(track_det[4])
            detec = patches.Rectangle((track_det[0], track_det[1]), (track_det[2]-track_det[0]),(track_det[3]-track_det[1]),
                                      linewidth=1.5, edgecolor=color, facecolor='none')
            ax1.add_patch(detec)

            length = 10
            if np.abs(vx) + np.abs(vy) > 0.5:
                ax1.arrow(x, y, vx*length, vy*length, head_width=10, head_length=10, fc=color, ec=color)

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

    gt_list = Video(Video().getgroundTruthown(gt, begin, end))

    compute_IDF1(video_detections, gt_list)

    print(compute_IDF1(video_detections, gt_list))

    return video_detections


def tracking_gt(detec_dir, gt, begin, end, num_color,  visualize=True):

    detections_list = Video(Video().getgroundTruthown(gt, begin, end))  # 450

    cmap = get_cmap(num_color)

    ax1 = plt.subplot()
    plt.ion()
    video_detections = []

    for frame in detections_list.list_frames:
        print(frame.frame_id)
        detections = []
        frame_num = frame.frame_id
        for box_detec in frame.bboxes:
            boxes = np.array([box_detec.top_left[0], box_detec.top_left[1], box_detec.width + box_detec.top_left[0],
                              box_detec.height + box_detec.top_left[1]])

            detections.append(boxes)

        frame_detections = []

        path = '../datasets/train/S03/c010/frames/image' + str(frame_num) + '.jpg'
        path = "/home/marc/M6/M6_VA/week_3/aic19-track1-mtmc-train/train/S01/c001/frames/image_" + str(frame_num).zfill(3) + ".png"

        im = np.array(Image.open(path), dtype=np.uint8)

        for gt_bbox in frame.bboxes:
            print(gt_bbox)
            id = gt_bbox.det_id

            bbox = np.array([gt_bbox.top_left[0], gt_bbox.top_left[1], gt_bbox.top_left[0]+gt_bbox.width, gt_bbox.top_left[1]+gt_bbox.height])

            track_det = np.concatenate((bbox, [id])).astype(np.uint64)

            frame_detections.append([track_det[4], track_det[0], track_det[1], (track_det[2] - track_det[0]),
                                     (track_det[3] - track_det[1])])

            color = cmap(1)
            color = "green"
            detec = patches.Rectangle((track_det[0], track_det[1]), (track_det[2]-track_det[0]),(track_det[3]-track_det[1]),
                                      linewidth=1.5, edgecolor=color, facecolor='none')
            ax1.add_patch(detec)

            caption_id = "ID: {}".format(track_det[4])
            ax1.text(track_det[0], track_det[1] - 10, caption_id, color=color)

            ax1.text(10, 60, "Frame: {}".format(frame_num), color='w', fontsize=11)

        video_detections.append(frame_detections)
        if visualize:
            ax1.axis('off')
            ax1.imshow(im)
            plt.show()
            plt.savefig("gt_tracking_images/image_{}".format(frame_num))
            plt.pause(0.001)
            ax1.clear()

    gt_list = Video(Video().getgroundTruthown(gt, begin, end))

    compute_IDF1(video_detections, gt_list)

    print(compute_IDF1(video_detections, gt_list))


#detec_dir = 'detections_mask_rcnn_fine_tune.txt'
#gt = 'annotation_only_cars.txt'
#tracking_kalman(detec_dir, gt, 0, 2141, 50,  visualize=True)