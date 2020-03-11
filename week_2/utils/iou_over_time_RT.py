from model.video import *
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")
from metrics.evaluation_funcs import *
from PIL import Image
from matplotlib import patches
import matplotlib


def iou_over_time_RT(gt_dir, detections):
    gt_video = Video(Video().getgroundTruthown(gt_dir, 391, 391 + 350))

    yolo_video = Video(Video().getgroundTruthown(detections, 391, 391 + 350))

    iou_by_frame_yolo = iou_overtime(gt_video, yolo_video, thres=0.5)

    num_framesyolo = len(iou_by_frame_yolo)

    x = []
    fig = plt.figure()
    ax = plt.subplot(212)
    Ln, = ax.plot(iou_by_frame_yolo)
    ax.set_xlim([0, 350])
    ax.set_xlabel('frames')
    ax.set_ylabel('IOU')
    ax1 = plt.subplot(211)
    plt.ion()

    for i in range(num_framesyolo):
        x.append(iou_by_frame_yolo[i])
        Ln.set_ydata(x)
        Ln.set_xdata(range(len(x)))

        detections_bboxes = yolo_video.get_frame_by_id(391 + i)

        gt_bboxes = gt_video.get_frame_by_id(391 + i)
        path = 'datasets/train/S03/c010/frames/image' + str(391 + i) + '.jpg'

        im = np.array(Image.open(path), dtype=np.uint8)
        ax1.imshow(im)

        iouframe, TP, FP, FN = iou_frame(detections_bboxes, gt_bboxes, thres=0.5)

        for bbox in gt_bboxes.bboxes:
            ground = patches.Rectangle(bbox.top_left,
                                     bbox.width, bbox.height,
                                     linewidth=1.75, edgecolor='g', facecolor='none',label='groundtruth')
            ax1.add_patch(ground)

        for bbox_noisy in detections_bboxes.bboxes:
            bb = bbox_noisy.to_result()
            detec = patches.Rectangle(bbox_noisy.top_left,
                                     bbox_noisy.width, bbox_noisy.height,
                                     linewidth=1.5, edgecolor='r', facecolor='none',label='detections')

            ax1.add_patch(detec)


        plt.legend(handles=[ground,detec],loc="lower left",prop={'size': 6})
        ax1.axis('off')
        plt.title('IOU over time')
        plt.show()
        plt.pause(0.005)
        ax1.clear()
