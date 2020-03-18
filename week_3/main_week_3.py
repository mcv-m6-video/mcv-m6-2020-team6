from off_the_shelf import *
from fine_tunning import *
from tracking_kalman import *
from tracking_overlap import *


def task11(detector):
    """Object detection off-the-shelf"""

    if detector == 'yolo':
        yolo_off_the_shelf()

    elif detector == 'retinanet':
        retinanet_off_the_shelf()

    elif detector == 'mask_rcnn':
        mask_rcnn_off_the_shelf()


def task12():
    """Fine-tune to your data"""

    mask_rcnn_fine_tune()


def task21():
    """Tracking by overlap"""

    detec_dir = 'detections_mask_rcnn_fine_tune_25.txt'

    gt = 'annotation_only_cars.txt'

    tracking_overlap(detec_dir, gt, 0, 2141, 100, visualize=False)

    video_detections = tracking_overlap(detec_dir, gt, 450, 900, 50, visualize=True)

    #visualize_tracking(video_detections, 450, 900, save=True, title='tracking_overlap')


def task22():
    """Tracking with a Kalman Filter"""

    detec_dir = 'detections_mask_rcnn_fine_tune_25.txt'

    gt = 'annotation_only_cars.txt'

    #tracking_kalman(detec_dir, gt, 0, 2141, 50, visualize=False)

    tracks = tracking_kalman(detec_dir, gt, 430, 500, 50, visualize=False)

    visualize_tracking_kal(tracks, 450, 900, save=True, title='tracking_kalman')


def task23():
    """IDF1 for Multiple Object Tracking"""

    detec_dir = 'detections_mask_rcnn_fine_tune_25.txt'

    gt = 'annotation_only_cars.txt'

    tracking_kalman(detec_dir, gt, 0, 2141, 50, visualize=False)


def task25():
    """AI CITY CHALLENGE"""

    detec_dir = 'aic19-track1-mtmc-train/train/S01/c001/det/det_mask_rcnn.txt'

    gt = 'aic19-track1-mtmc-train/train/S01/c001/gt/gt.txt'

    tracking_kalman(detec_dir, gt, 1, 50, 50, visualize=True)


if __name__ == '__main__':
    # task11('retinanet')  # yolo, retinanet or mask_rcnn
    # task12()
    # task21()
    # task23()
    # task24()
    task25()