from math import ceil
from statistics import mean

import matplotlib.pyplot as plt
import motmetrics as mm
import numpy as np

from model.bbox import BBox
from model.frame import Frame
from model.video import Video


def performance_evaluation(TP, FN, FP):
    """
    performance_evaluation_window()

    Function to compute different performance indicators (Precision, accuracy, 
    sensitivity/recall) at the object level
    
    [precision, sensitivity, accuracy] = PerformanceEvaluationPixel(TP, FN, FP)
    
       Parameter name      Value
       --------------      -----
       'TP'                Number of True  Positive objects
       'FN'                Number of False Negative objects
       'FP'                Number of False Positive objects
    
    The function returns the precision, accuracy and sensitivity
    """

    precision = float(TP) / float(TP + FP)  # Q: What if i do not have TN?
    recall = float(TP) / float(TP + FN)
    f1_score = (2 * (precision * recall)) / (precision + recall)

    return [precision, recall, f1_score]


def iou_frame(detections_frame: Frame, gt_frames: Frame, thres):
    TP = 0
    FP = 0
    FN = 0
    iouframe = []
    u = 0
    i = 0

    for u in detections_frame.bboxes:
        ious = []
        """bboxes=detections_frames.get_bboxes()
        for i in bboxes:"""
        for i in gt_frames.bboxes:
            # print(u)
            # print(i)
            ious.append(iou_bbox_2(u, i))

        if ious:
            if max(ious) > thres:
                TP += 1
            iouframe.append(max(ious))
        else:
            iouframe.append(0)

    FP = len(detections_frame.bboxes) - TP

    FN = len(gt_frames.bboxes) - TP

    return iouframe, TP, FP, FN


def iou_video(gt: Video, detections: Video, thres=0.1):
    TP = 0
    iou_frm = []
    for i in detections.list_frames:
        frame_gt = gt.get_frame_by_id(i.frame_id)
        # print(frame_detec.bboxes)

        ioufrm, TP_fr, FP, FN = iou_frame(i, frame_gt, thres)
        TP += TP_fr
        iou_frm.append(ioufrm)
    return TP, iou_frm


def iou_TFTN_video(gt: Video, detections: Video, thres=0.1):
    TP = 0
    FP = 0
    FN = 0

    TP, iu = iou_video(gt, detections, thres)

    FP = len(detections.get_detections_all()) - TP

    FN = len(gt.get_detections_all()) - TP

    return TP, FP, FN


def iou_overtime(gt: Video, detections: Video, thres=0.1):
    iou_by_frame = []
    for i in detections.list_frames:
        iouframe, TP, FP, FN = iou_frame(i, gt.get_frame_by_id(i.frame_id), thres)

        if len(iouframe) > 1:
            iou_mean = mean(iouframe)
        else:
            if not iouframe:
                iou_mean = float(0)
                iou_by_frame.append(iou_mean)
            else:
                iou_mean = iouframe[0]

        iou_by_frame.append(iou_mean)

    return iou_by_frame


def iou_bbox_2(bboxA: BBox, bboxB: BBox):
    # compute the intersection over union of two bboxes

    # Format of the bboxes is [tly, tlx, bry, brx, ...], where tl and br
    # indicate top-left and bottom-right corners of the bbox respectively.

    # determine the coordinates of the intersection rectangle
    xA = max(bboxA.top_left[0], bboxB.top_left[0])
    yA = max(bboxA.top_left[1], bboxB.top_left[1])
    xB = min(bboxA.get_bottom_right()[0], bboxB.get_bottom_right()[0])
    yB = min(bboxA.get_bottom_right()[1], bboxB.get_bottom_right()[1])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both bboxes
    bboxAArea = (bboxA.get_bottom_right()[1] - bboxA.top_left[1] + 1) * (
        bboxA.get_bottom_right()[0] - bboxA.top_left[0] + 1
    )
    bboxBArea = (bboxB.get_bottom_right()[1] - bboxB.top_left[1] + 1) * (
        bboxB.get_bottom_right()[0] - bboxB.top_left[0] + 1
    )

    iou = interArea / float(bboxAArea + bboxBArea - interArea)

    # return the intersection over union value
    return iou


def iou_map(Bbox_detec: BBox, gt_frames: Frame, thres):
    TP = 0
    FP = 0
    iouframe = []
    ious = []

    for i in gt_frames.bboxes:
        ious.append(iou_bbox_2(Bbox_detec, i))
        if ious:
            if ious[-1] > thres:
                TP += 1
                iouframe.append(ious[-1])
        else:
            iouframe.append(0)
    if all(v == 0 for v in iouframe) or not iouframe:
        FP = 1

    return iouframe, TP, FP


def mAP(gt: Video, detections: Video, fname):
    TP = 0
    FP = 0

    bbox_gt = gt.get_detections_all()
    bbox_detected = detections.get_detections_all()
    bbox_detected.sort(key=lambda x: x.confidence, reverse=True)
    precision = []
    recall = []
    threshold = ceil((1 / len(bbox_detected)) * 10) / 10
    step_precision = []
    checkpoint = 0
    j = 0

    for i in bbox_detected:
        j = +1
        gtframe = gt.get_frame_by_id(i.get_frame_id())
        [iouframe, TPbb, FPbb] = iou_map(i, gtframe, 0.5)

        TP += TPbb

        FP += FPbb

        # Save metrics
        precision.append(TP / (TP + FP))
        recall.append(TP / len(bbox_gt))

    for step in np.linspace(0.1, 1.1, 11):
        if step <= recall[-1] + 0.1:
            index_0 = min(
                range(len(recall)), key=lambda i: abs(recall[i] - (step - 0.1))
            )
            index_1 = min(range(len(recall)), key=lambda i: abs(recall[i] - step))
            step_precision.append(max(precision[index_0:index_1]))

    """
    if recall[-1] > threshold or j == len(bbox_detected) - 1:
        step_precision.append(max(precision[checkpoint:len(precision) - 2]))

        checkpoint = len(precision)
        threshold += 0.1
    """
    mAP = sum(step_precision) / 11
    print("mAP: {}\n".format(mAP))
    plot_precision_recall_11_point(precision, recall, step_precision, fname)


def precision_recall_ious(gt: Video, detections: Video, fname):
    PRECISION = []
    RECALL = []
    MAP = []
    for u in range(0, 10):
        TP = 0
        FP = 0

        bbox_gt = gt.get_detections_all()
        bbox_detected = detections.get_detections_all()
        bbox_detected.sort(key=lambda x: x.confidence, reverse=True)
        precision = []
        recall = []
        step_precision = []
        j = 0
        step = u * 0.05
        for i in bbox_detected:
            j = +1
            gtframe = gt.get_frame_by_id(i.get_frame_id())
            [iouframe, TPbb, FPbb] = iou_map(i, gtframe, 0.5 + step)

            TP += TPbb
            FP += FPbb

            # Save metrics
            precision.append(TP / (TP + FP))
            recall.append(TP / len(bbox_gt))
        """
        for step in np.linspace(0.1, 1.1, 11):
            if step <= recall[-1] + 0.1:
                index_0 = min(range(len(recall)), key=lambda i: abs(recall[i] - (step - 0.1)))
                index_1 = min(range(len(recall)), key=lambda i: abs(recall[i] - step))
                step_precision.append(max(precision[index_0:index_1]))
                mAP = sum(step_precision) / 11
        """
        """
        if recall[-1] > threshold or j == len(bbox_detected) - 1:
            step_precision.append(max(precision[checkpoint:len(precision) - 2]))

            checkpoint = len(precision)
            threshold += 0.1
        """

        PRECISION.append(precision)
        RECALL.append(recall)
        # MAP.append(mAP)
    plot_precision_recall_ious(PRECISION, RECALL, fname)


"""
def plot_precision_recall_curve(precision, recall):
    # Data for plotting
    fig, ax = plt.subplots()
    ax.plot(recall, precision)

    ax.set(xlabel='Recall', ylabel='Precision',
           title='Precision-Recall Curve')
    ax.grid()

    fig.savefig("precision-recall.png")
    # plt.show()

    precisions = np.array(precision)
    recalls = np.array(recall)
    prec_at_rec = []
    for recall_level in np.linspace(0.0, 1.0, 11):
        try:
            args = np.argwhere(recalls >= recall_level).flatten()
            prec = max(precisions[args])
        except ValueError:
            prec = 0.0
        prec_at_rec.append(prec)

    avg_prec = np.mean(prec_at_rec)
"""


def plot_precision_recall_11_point(precision, recall, step_precision, fname):
    precisionValues = step_precision
    plt.plot(recall, precision, label="Precision")
    recallValues = np.linspace(0, 1, 11)
    for j in range(11 - len(precisionValues)):
        precisionValues.append(0)
    plt.plot(
        recallValues, step_precision, "or", label="11-point interpolated precision"
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(shadow=True)
    plt.grid()
    plt.savefig(fname)
    # plt.show()
    plt.clf()


def plot_precision_recall_ious(precision, recall, fname):
    for iou_t in range(0, 10):
        iou_step = 0.5 + (iou_t * 0.05)
        # plt.plot(recall[iou_t], precision[iou_t], label='P-R Thres:' + '{0:.2f}'.format(iou_step)+'(mAP:''{0:.2f}'.format(map[iou_t])+')')
        plt.plot(
            recall[iou_t],
            precision[iou_t],
            label="P-R Thres:" + "{0:.2f}".format(iou_step),
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Precision-Recall Curve")
    plt.legend(shadow=True)
    plt.grid()
    plt.savefig(fname)
    # plt.show()
    plt.clf()


def compute_IDF1(trackings, gt_boxes):

    acc = mm.MOTAccumulator(auto_id=True)

    for gt_frame, det_frame in zip(gt_boxes.list_frames, trackings):
        mm_det_bboxes = []
        mm_gt_bboxes = []
        det_ids = []
        gt_ids = []

        for gt_bbox in gt_frame.bboxes:

            gt_ids.append(gt_bbox.det_id)

            mm_gt_bboxes.append(
                [
                    gt_bbox.top_left[0],
                    gt_bbox.top_left[1],
                    gt_bbox.width,
                    gt_bbox.height,
                ]
            )

        for det_bbox in det_frame:
            print(det_bbox[0])

            mm_det_bboxes.append([det_bbox[1], det_bbox[2], det_bbox[3], det_bbox[4]])
            det_ids.append(det_bbox[0])

        distances_gt_det = mm.distances.iou_matrix(
            mm_gt_bboxes, mm_det_bboxes, max_iou=1.0
        )
        acc.update(gt_ids, det_ids, distances_gt_det)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['idf1', 'idp', 'idr', 'recall', 'precision', 'motp', 'mota'], name='score:')
    print(summary)

    return summary


def compute_IDF1_2(trackings, gt_boxes, ini, fin):
    trackings = [i for i in trackings if i]
    acc = mm.MOTAccumulator(auto_id=True)
    for i in range(ini, fin):
        gt_frame = gt_boxes.get_frame_by_id(i)
        mm_det_bboxes = []
        mm_gt_bboxes = []
        det_ids = []
        gt_ids = []
        if gt_frame != Frame():
            for gt_bbox in gt_frame.bboxes:

                gt_ids.append(gt_bbox.det_id)

                mm_gt_bboxes.append(
                    [
                        gt_bbox.top_left[0],
                        gt_bbox.top_left[1],
                        gt_bbox.width,
                        gt_bbox.height,
                    ]
                )
        else:
            gt_ids.append(None)
            mm_gt_bboxes.append([None, None, None, None])

        det_frame = list(filter(lambda x: x[0][6] == i, trackings))
        if det_frame != []:
            for det_bbox in det_frame[0]:
                mm_det_bboxes.append(
                    [det_bbox[1], det_bbox[2], det_bbox[3], det_bbox[4]]
                )
                det_ids.append(det_bbox[0])
        else:
            det_ids.append(None)
            mm_det_bboxes.append([None, None, None, None])

        distances_gt_det = mm.distances.iou_matrix(
            mm_gt_bboxes, mm_det_bboxes, max_iou=1.0
        )
        acc.update(gt_ids, det_ids, distances_gt_det)
        np.warnings.filterwarnings("ignore")

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['idf1', 'idp', 'idr', 'recall', 'precision', 'motp', 'mota'], name='score:')
    print(summary)

    return summary


def compute_IDF1_overlap(trackings, gt_boxes, ini, fin):
    trackings = [i for i in trackings if i]
    acc = mm.MOTAccumulator(auto_id=True)
    for i in range(ini, fin):
        gt_frame = gt_boxes.get_frame_by_id(i)
        mm_det_bboxes = []
        mm_gt_bboxes = []
        det_ids = []
        gt_ids = []
        if len(gt_frame.bboxes) != 0:
            for gt_bbox in gt_frame.bboxes:

                gt_ids.append(gt_bbox.det_id)

                mm_gt_bboxes.append(
                    [
                        gt_bbox.top_left[0],
                        gt_bbox.top_left[1],
                        gt_bbox.width,
                        gt_bbox.height,
                    ]
                )
        else:
            gt_ids.append(None)
            mm_gt_bboxes.append([None, None, None, None])

        det_frame = list(filter(lambda x: x[0][5] == i, trackings))
        if det_frame != []:
            for det_bbox in det_frame[0]:
                mm_det_bboxes.append(
                    [det_bbox[1], det_bbox[2], det_bbox[3], det_bbox[4]]
                )
                det_ids.append(det_bbox[0])
        else:
            det_ids.append(None)
            mm_det_bboxes.append([None, None, None, None])

        distances_gt_det = mm.distances.iou_matrix(
            mm_gt_bboxes, mm_det_bboxes, max_iou=1.0
        )
        acc.update(gt_ids, det_ids, distances_gt_det)
        np.warnings.filterwarnings("ignore")

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['idf1', 'idp', 'idr', 'recall', 'precision','motp', 'mota'], name='score:')
    print(summary)

    return summary


def compute_IDF1_voverlap(trackings, gt_boxes, ini, fin):

    acc = mm.MOTAccumulator(auto_id=True)
    for i in range(ini, fin):
        gt_frame = gt_boxes.get_frame_by_id(i)
        mm_det_bboxes = []
        mm_gt_bboxes = []
        det_ids = []
        gt_ids = []
        if len(gt_frame.bboxes) != 0:
            for gt_bbox in gt_frame.bboxes:

                gt_ids.append(gt_bbox.det_id)

                mm_gt_bboxes.append(
                    [
                        gt_bbox.top_left[0],
                        gt_bbox.top_left[1],
                        gt_bbox.width,
                        gt_bbox.height,
                    ]
                )
        else:
            gt_ids.append(None)
            mm_gt_bboxes.append([None, None, None, None])

        det_frame = trackings.get_frame_by_id(i)
        if len(det_frame.bboxes) != 0:
            for det_bbox in det_frame.bboxes:
                mm_det_bboxes.append(
                    [
                        det_bbox.top_left[0],
                        det_bbox.top_left[1],
                        det_bbox.width,
                        det_bbox.height,
                    ]
                )
                det_ids.append(det_bbox.det_id)
        else:
            det_ids.append(None)
            mm_det_bboxes.append([None, None, None, None])

        distances_gt_det = mm.distances.iou_matrix(
            mm_gt_bboxes, mm_det_bboxes)
        acc.update(gt_ids, det_ids, distances_gt_det)

        np.warnings.filterwarnings("ignore")

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['idf1', 'idp', 'idr', 'recall', 'precision', 'motp', 'mota'], name='score:')
    print(summary)

    return summary
