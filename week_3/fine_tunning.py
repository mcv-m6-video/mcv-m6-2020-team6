from week_3.model.read_annotation_old import read_annotations
from week_3.utils.iou_over_time_RT import *
from week_3.metrics.evaluation_funcs import *
from week_3.model.video import Video


def mask_rcnn_fine_tune():
    # file = 'm6-full_annotation.xml'
    # read_annotations(file)

    gt_dir = 'annotation_only_cars.txt'

    detec_dir = 'detections_mask_rcnn_fine_tune_40.txt'

    gt_video = Video(Video().getgroundTruthown(gt_dir, 0, 2141))

    detections = Video(Video().getgroundTruthown(detec_dir, 0, 2141))

    TP1, FP1, FN1 = iou_TFTN_video(gt_video, detections, thres=0.5)

    [precision, recall, f1_score] = performance_evaluation(TP1, FN1, FP1)

    print("\nMask R-CNN fine-tune:"
          "\nPrecision:", precision,
          "\nF1_score:", f1_score,
          "\nRecall:", recall)

    mAP(gt_video, detections, fname='precision_recall_mask_rcnn_fine_tune.png')

    #iou_over_time_RT_mask_rcnn(gt_dir, detec_dir, 391, 855)
    #detection_visualization(gt_dir, detec_dir, 391, 855, name_title='Mask R-CNN fine-tune')
    #visualize_detections_with_iou(detec_dir, gt_dir, 700, 1000, save=True, title='mask_rcnn_fine_tune_iou')
    visualize_detections(detec_dir, gt_dir, 700, 1000, save=True, title='Mask R-CNN fine-tune')


# mask_rcnn_fine_tune()
