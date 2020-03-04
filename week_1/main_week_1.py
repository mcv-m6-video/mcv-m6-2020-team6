from week_1.utils.iou_over_time_RT import iou_over_time_RT
from week_1.metrics.evaluation_funcs import *
from week_1.model.read_annotation import read_annotations
from week_1.utils.show_bb_singleFrame import *
from week_1.metrics.Optical_flow_metrics import pepn, msen, Flow_read
from week_1.metrics.Optical_flow_visualization import *


def task0():
    """Conversion cvat xml to ai city challenge format """
    file = 'm6-full_annotation.xml'
    read_annotations(file)


def task11(visualization,gt_modif):
    """Implement IoU and mAP:
    -Add noise to change size and position of bounding boxes
    -Add probability to generate/delete bounding boxes
    +Analysis & Evaluation"""

    # gt_dir = '../datasets/train/S03/c010/gt/gt.txt'
    gt_dir = 'annotation.txt'

    # Apply modifications and eliminate samples of the gt given.
    gt_video = Video(Video().getgroundTruthown(gt_dir, 0, 2141))
    gt_video_modif1 = Video(Video().getgroundTruthown(gt_dir, 0, 2141))
    gt_video_modif2 = Video(Video().getgroundTruthown(gt_dir, 0, 2141))
    # Apply modifications and eliminate samples of the gt given

    # First modification:
    # modify randomnly the bounding boxes by 40%
    gt_video_modif1.modify_random_bboxes(0.4)
    # eliminate the randomnly 10% of the bounding boxes
    gt_video_modif1.eliminate_random_bboxes(0.1)

    # Second modification:
    # modify randomnly the bounding boxes by 20%
    gt_video_modif2.modify_random_bboxes(0.2)
    # eliminate the randomnly 70% of the bounding boxes
    gt_video_modif2.eliminate_random_bboxes(0.7)




    # IOU global
    TP1, FP1, FN1 = iou_TFTN_video(gt_video, gt_video_modif1, thres=0.5)
    [precision1, recall1, f1_score1] = performance_evaluation(TP1, FN1, FP1)


    TP2, FP2, FN2 = iou_TFTN_video(gt_video, gt_video_modif2, thres=0.5)
    [precision2, recall2, f1_score2] = performance_evaluation(TP2, FN2, FP2)


    # Evaluation
    print("\nFist modification:"
          "\nPrecision:", precision1,
          "\nF1_score:", f1_score1,
          "\nRecall:", recall1)
    mAP(gt_video, gt_video_modif1, fname='precision_recall_11_interp_gt_video_modif1.png')
    precision_recall_ious(gt_video, gt_video_modif1, fname='precision_recall_ious_modified1.png')

    print("\nSecond modification:"
          "\nPrecision:", precision2,
          "\nF1_score:", f1_score2,
          "\nRecall:", recall2)
    mAP(gt_video, gt_video_modif2, fname='precision_recall_11_interp_gt_video_modif2.png')
    precision_recall_ious(gt_video, gt_video_modif2, fname='precision_recall_ious_modified2.png')

    # IoU visualization

    if gt_modif == 'gt_modif1': gt_video_modif=gt_video_modif1
    elif gt_modif == 'gt_modif2': gt_video_modif=gt_video_modif2

    path = '../datasets/train/S03/c010/frames/image391.jpg'
    detections_bboxes = gt_video_modif.get_frame_by_id(391)
    gt_bboxes = gt_video.get_frame_by_id(391)

    iouframe, TP, FP, FN = iou_frame(detections_bboxes, gt_bboxes, thres=0.5)

    for i in iouframe:
        print('\n IOU:', i)

    # IoU visualization
    if (visualization):
        show_bboxes(path, gt_bboxes, detections_bboxes)


def task112(gt_modif):
    """mAP using the modified groundtruth"""

    # gt_dir = '../datasets/train/S03/c010/gt/gt.txt'
    gt_dir = 'annotation.txt'

    gt_video = Video(Video().getgroundTruthown(gt_dir, 0, 2141))

    if gt_modif == 'gt_modif1':
        # First modification:
        gt_video_modif = Video(Video().getgroundTruthown(gt_dir, 0, 2141))
        gt_video_modif.modify_random_bboxes(0.4)
        gt_video_modif.eliminate_random_bboxes(0.1)

    elif gt_modif == 'gt_modif2':
        # Second modification:
        gt_video_modif = Video(Video().getgroundTruthown(gt_dir, 0, 2141))
        gt_video_modif.modify_random_bboxes(0.2)
        gt_video_modif.eliminate_random_bboxes(0.7)

    # mAP(mean average) using mAP@50
    mAP(gt_video, gt_video_modif, fname='precision_recall_11_interp_modified.png')
    mAP_tensorflow(gt_video, gt_video_modif)
    precision_recall_ious(gt_video, gt_video_modif, fname='precision_recall_ious_modified.png')


def task12(detections):
    """mAP using provided predictions"""

    # gt_dir = '../datasets/train/S03/c010/gt/gt.txt'
    gt_dir = 'annotation.txt'

    gt_video = Video(Video().getgroundTruthown(gt_dir, 0, 2141))

    if detections == 'mask_rcnn':
        detect = '../datasets/train/S03/c010/det/det_mask_rcnn.txt'
        detections_video = Video(Video().getgroundTruthown(detect, 0, 2141))
    elif detections == 'ssd512':
        detect = '../datasets/train/S03/c010/det/det_ssd512.txt'
        detections_video = Video(Video().getgroundTruthown(detect, 0, 2141))
    elif detections == 'yolo':
        detect = '../datasets/train/S03/c010/det/det_yolo3.txt'
        detections_video = Video(Video().getgroundTruthown(detect, 0, 2141))

    mAP(gt_video, detections_video, fname='precision_recall_11_interp_'+detections+'.png')
    mAP_tensorflow(gt_video, detections_video)
    precision_recall_ious(gt_video, detections_video, fname='precision_recall_ious_detections_'+detections+'.png')


def task2():
    """Temporal analysis: IOU overtime """
    # gt_dir = 'datasets/train/S03/c010/gt/gt.txt'
    gt_dir = 'annotation.txt'

    yolo = '../datasets/train/S03/c010/det/det_yolo3.txt'
    ssd = '../datasets/train/S03/c010/det/det_ssd512.txt'
    rcnn = '../datasets/train/S03/c010/det/det_mask_rcnn.txt'

    gt_video = Video(Video().getgroundTruthown(gt_dir,  0, 2141))
    #gt_video = Video(Video().getgroundTruthown(gt_dir, 391, 391 + 350))

    gt_video_modif1 = Video(Video().getgroundTruthown(gt_dir, 0, 2141))
    #gt_video_modif1 = Video(Video().getgroundTruthown(gt_dir, 391, 391 + 350))
    gt_video_modif2 = Video(Video().getgroundTruthown(gt_dir, 391, 391 + 350))

    # First modification:
    # modify randomnly the bounding boxes by 1%
    gt_video_modif1.modify_random_bboxes(0.1)

    # iou-gt
    iou_by_frame = iou_overtime(gt_video_modif1, gt_video, thres=0.5)
    num_frames = len(iou_by_frame)
    plt.plot(iou_by_frame)
    plt.ylabel('IOU')
    plt.xlabel('Frames')
    plt.title('IOU-overtime:GT modified')
    axes = plt.gca()
    axes.set_ylim(0, 1)
    axes.set_xlim(0, num_frames)
    plt.savefig('IOU-overtime:GT modified.png')
    plt.show()

    # iou_overtime - YOLO
    yolo_video = Video(Video().getgroundTruthown(yolo, 0, 2141))
    #yolo_video = Video(Video().getgroundTruthown(yolo, 391, 391 + 350))
    iou_by_frame_yolo = iou_overtime(gt_video, yolo_video, thres=0.5)
    num_framesyolo = len(iou_by_frame_yolo)
    plt.plot(iou_by_frame_yolo)
    plt.ylabel('IOU')
    plt.xlabel('Frames')
    plt.title('IOU-overtime:YOLO')
    axes = plt.gca()
    axes.set_ylim(0, 1)
    axes.set_xlim(0, num_framesyolo)
    plt.savefig('IOU-overtime:YOLO.png')
    plt.show()

    # iou_overtime - ssd512
    ssd_video = Video(Video().getgroundTruthown(ssd, 0, 2141))
    #ssd_video = Video(Video().getgroundTruthown(ssd, 391, 391 + 350))
    iou_by_frame_ssd = iou_overtime(gt_video, ssd_video, thres=0.5)
    num_framesssd = len(iou_by_frame_ssd)
    plt.plot(iou_by_frame_ssd)
    plt.ylabel('IOU')
    plt.xlabel('Frames')
    plt.title('IOU-overtime:SSD')
    axes = plt.gca()
    axes.set_ylim(0, 1)
    axes.set_xlim(0, num_framesssd)
    plt.savefig('IOU-overtime:SSD.png')
    plt.show()

    # iou_overtime - mask_rccn
    rcnn_video = Video(Video().getgroundTruthown(rcnn, 0, 2141))
    #rcnn_video = Video(Video().getgroundTruthown(rcnn, 391, 391 + 350))
    iou_by_frame_rcnn = iou_overtime(gt_video, rcnn_video, thres=0.5)
    num_framesrcnn = len(iou_by_frame_rcnn)
    plt.plot(iou_by_frame_rcnn)
    plt.ylabel('IOU')
    plt.xlabel('Frames')
    plt.title('IOU-overtime:RCNN')
    axes = plt.gca()
    axes.set_ylim(0, 1)
    axes.set_xlim(0, num_framesrcnn)
    plt.savefig('IOU-overtime:RCNN.png')
    plt.show()

    # IoU over time - All CNNs
    plt.plot(iou_by_frame_ssd, color="navy", label="SSD")
    plt.plot(iou_by_frame_rcnn, color="orchid", label="Mask R-CNN")
    plt.plot(iou_by_frame_yolo, color="turquoise", label="YOLO")
    plt.ylabel('IOU')
    plt.xlabel('Frames')
    plt.title('IoU over time: SSD vs Mask R-CNN vs YOLO')
    axes = plt.gca()
    axes.set_ylim(0, 1)
    axes.set_xlim(0, num_framesrcnn)
    plt.legend()
    plt.savefig('IOU-overtime:All.png')
    plt.show()

    # Real Time IoU over time using YOLO detections
    iou_over_time_RT(gt_dir, yolo)


def task3(seq):
    """Optical flow: Numerical result for MSEN and PEPN, histogram error and error visualization"""

    # change the number of sequence to visualize the results for both cases
    if seq == 45:
        gt_dir = "../datasets/kitti/groundtruth/000045_10.png"
        test_dir = '../datasets/kitti/results/LKflow_000045_10.png'
    if seq == 157:
        gt_dir = "../datasets/kitti/groundtruth/000157_10.png"
        test_dir = '../datasets/kitti/results/LKflow_000157_10.png'

    flow_gt, F_gt = Flow_read(gt_dir)
    flow_test, F_test = Flow_read(test_dir)

    MSEN = msen(F_gt, F_test)
    PEPN = pepn(F_gt, F_test, 3)

    print(MSEN)
    print(PEPN)


def task4(option):
    flow_dir_0 = "../datasets/kitti/groundtruth/000045_10.png"
    flow_dir_1 = "../datasets/kitti/groundtruth/000157_10.png"
    # flow_dir_2 = "../datasets/kitti/results/LKflow_000045_10.png"
    # flow_dir_3 = "../datasets/kitti/results/LKflow_000157_10.png"
    img_dir_0 = "../datasets/kitti/images/000045_10.png"
    img_dir_1 = "../datasets/kitti/images/000157_10.png"
    # img_dir_2 = "../datasets/kitti/images/000045_10.png"
    # img_dir_3 = "../datasets/kitti/images/000157_10.png"

    if option == 'color_based':
        OF_visualization(flow_dir_0, fname_output='flow_gt_45_color.png', maxflow=None)
        OF_visualization(flow_dir_1, fname_output='flow_gt_157_color.png', maxflow=None)
        # OF_visualization(flow_dir_2, fname_output='flow_res_45_color.png', maxflow=None)
        # OF_visualization(flow_dir_3, fname_output='flow_res_157_color.png', maxflow=None)
    elif option == 'quiver_based':
        OF_quiver_visualization(img_dir_0, flow_dir_0, step=15, fname_output='flow_gt_45_quiver.png')
        OF_quiver_visualization(img_dir_1, flow_dir_1, step=15, fname_output='flow_gt_157_quiver.png')
        # OF_quiver_visualization(img_dir_2, flow_dir_2, step=8, fname_output='flow_res_45_quiver.png')
        # OF_quiver_visualization(img_dir_3, flow_dir_3, step=8, fname_output='flow_res_147_quiver.png')


if __name__ == '__main__':
    # task0()
    task11(False,'gt_modif1')
    # task112('gt_modif1')  # 'gt_modif1' or 'gt_modif2'
    # task12('ssd512') # mask_rcnn, yolo or ssd512
    # task2()
    # task3(157)  #45 or 157
    # task4('color_based')  # 'color_based' or 'quiver_based'
