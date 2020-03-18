from week_3.model.read_annotation_old import read_annotations
from week_3.utils.iou_over_time_RT import *
from week_3.metrics.evaluation_funcs import *
from week_3.model.video import Video


def yolo_off_the_shelf():
    # file = 'm6-full_annotation.xml'
    # read_annotations(file)

    gt_dir = 'annotation_only_cars.txt'

    detec_dir = 'detections_yolo.txt'

    gt_video = Video(Video().getgroundTruthown(gt_dir, 0, 2141))

    detections = Video(Video().getgroundTruthown(detec_dir, 0, 2141))

    TP1, FP1, FN1 = iou_TFTN_video(gt_video, detections, thres=0.5)

    [precision, recall, f1_score] = performance_evaluation(TP1, FN1, FP1)

    print("\nYolo off-the-shelf:"
          "\nPrecision:", precision,
          "\nF1_score:", f1_score,
          "\nRecall:", recall)

    mAP(gt_video, detections, fname='precision_recall_YOLO_v3.png')

    #iou_over_time_RT_yolo(gt_dir, detec_dir, 391, 855)

    visualize_detections(detec_dir, gt_dir, 700, 1000, save=True, title='yolo_off_the_shelf')
    #visualize_detections_with_iou(detec_dir, gt_dir, 700, 1000, save=True, title='yolo_off_the_shelf_iou')


def retinanet_off_the_shelf():
    # file = 'm6-full_annotation.xml'
    # read_annotations(file)

    gt_dir = 'annotation_only_cars.txt'

    detec_dir = 'detections_retinanet.txt'

    gt_video = Video(Video().getgroundTruthown(gt_dir, 0, 2141))

    detections = Video(Video().getgroundTruthown(detec_dir, 0, 2141))

    TP1, FP1, FN1 = iou_TFTN_video(gt_video, detections, thres=0.5)

    [precision, recall, f1_score] = performance_evaluation(TP1, FN1, FP1)

    print("\nRetinanet off-the-shelf:"
          "\nPrecision:", precision,
          "\nF1_score:", f1_score,
          "\nRecall:", recall)

    mAP(gt_video, detections, fname='precision_recall_retinanet.png')

    #iou_over_time_RT_retinanet(gt_dir, detec_dir, 391, 855)

    visualize_detections(detec_dir, gt_dir, 700, 1000, save=True, title='retinanet_off_the_shelf')
    #visualize_detections_with_iou(detec_dir, gt_dir, 700, 1000, save=True, title='retinanet_off_the_shelf_iou')


def mask_rcnn_off_the_shelf():
    # file = 'm6-full_annotation.xml'
    # read_annotations(file)

    gt_dir = 'annotation_only_cars.txt'

    detec_dir = 'detections_mask_rcnn.txt'

    gt_video = Video(Video().getgroundTruthown(gt_dir, 0, 2141))

    detections = Video(Video().getgroundTruthown(detec_dir, 0, 2141))

    TP1, FP1, FN1 = iou_TFTN_video(gt_video, detections, thres=0.5)

    [precision, recall, f1_score] = performance_evaluation(TP1, FN1, FP1)

    print("\nMask R-CNN off-the-shelf:"
          "\nPrecision:", precision,
          "\nF1_score:", f1_score,
          "\nRecall:", recall)

    mAP(gt_video, detections, fname='precision_recall_mask_rcnn.png')

    #iou_over_time_RT_mask_rcnn(gt_dir, detec_dir, 391, 855)
    #detection_visualization(gt_dir, detec_dir, 391, 855,'Mask R-CNN off-the-shelf')

    visualize_detections(detec_dir, gt_dir, 700, 1000, save=True, title='mask_rcnn_off_the_shelf')
    #visualize_detections_with_iou(detec_dir, gt_dir, 700, 1150, save=True, title='mask_rcnn_off_the_shelf_iou')


# yolo_off_the_shelf()
#retinanet_off_the_shelf()
#mask_rcnn_off_the_shelf()
