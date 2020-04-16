import gc
import os
import pickle

from metrics.evaluation_funcs import compute_IDF1_2, compute_IDF1_overlap
from model.video import Video
from mtsc.backgroundSubstration import background_adaptive_gaussian, get_frames, background_subtraction_sota
from mtsc.tracking_kalman_mtsc import tracking_kalman
from mtsc.tracking_overlap_mtsc import tracking_overlap


def case1(cam_list):
    """Adaptive Background Substration + Kalman S03 """
    # S03-C010
    path = 'datasets/aic19-track1-mtmc-train/train/S03/'
    print('Case 1: Adaptive+Kalman')

    while cam_list:
        cam = cam_list[0]
        if cam == 'c010': ini, fin = 536, 2141
        if cam == 'c011': ini, fin = 571, 2279
        if cam == 'c012': ini, fin = 607, 2422
        if cam == 'c013': ini, fin = 605, 2415
        if cam == 'c014': ini, fin = 584, 2332
        if cam == 'c015': ini, fin = 483, 1928
        print(ini)
        data_path = path + cam + '/frames/'
        roi_path = path + cam + '/roi.jpg'
        gt_dir = path + cam + '/gt/gt.txt'
        filename = 'mtsc/detections_bg/detections_bgadapt_' + cam + '.pkl'
        detections_list = background_adaptive_gaussian(data_path, roi_path, gt_dir, filename, cam=cam, num_frame=ini)
        [train_list, test_list] = get_frames(data_path, 0.25)
        gt_video = Video(Video().getgt(gt_dir, ini, fin))
        filename = 'mtsc/detections_bg/case1/results_bgadaptive_kalman_' + cam + '.pkl'
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                video_detections = pickle.load(file)
        else:
            video_detections = tracking_kalman(detections_list, path=path, begin=ini, end=fin, cam=cam, num_color=50,
                                               visualize=False, first_time=False)
            with open(filename, 'wb') as f:
                pickle.dump(video_detections, f)

        print('S03-' + cam + ' results:')
        print(compute_IDF1_2(video_detections, gt_video, ini, fin))
        gt_video = []
        video_detections = []
        detections_list = []
        gc.collect()
        cam_list.pop(0)


def case2(cam_list):
    """Adaptive Background Substration + Overlap S03 """
    # S03-C010
    path = 'datasets/aic19-track1-mtmc-train/train/S03/'
    print('Case 2: Adaptive+Overlap')

    while cam_list:
        cam = cam_list[0]
        cam_list.pop(0)
        if cam == 'c010': ini, fin = 536, 2141
        if cam == 'c011': ini, fin = 571, 2279
        if cam == 'c012': ini, fin = 607, 2422
        if cam == 'c013': ini, fin = 605, 2415
        if cam == 'c014': ini, fin = 584, 2332
        if cam == 'c015': ini, fin = 483, 1928

        data_path = path + cam + '/frames/'
        roi_path = path + cam + '/roi.jpg'
        gt_dir = path + cam + '/gt/gt.txt'
        filename = 'mtsc/detections_bg/detections_bgadapt_' + cam + '.pkl'
        detections_list = background_adaptive_gaussian(data_path, roi_path, gt_dir, filename, cam=cam, num_frame=ini)
        gt_video = Video(Video().getgt(gt_dir, ini, fin))
        # [train_list, test_list] = get_frames(data_path, 0.25)
        # visualize_mask2(detections_list, gt_video, ini, fin, test_list, 'bg_overlap')
        filename = 'mtsc/detections_bg/case2/results_bgadaptive_overlap_' + cam + '.pkl'
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                video_detections = pickle.load(file)
        else:

            video_detections = tracking_overlap(detections_list,fin, path, cam, num_color=50, visualize=False)
            with open(filename, 'wb') as f:
                pickle.dump(video_detections, f)


        print('S03-' + cam + ' results:')
        print(compute_IDF1_overlap(video_detections, gt_video, ini, fin))
        # visualize_tracking(video_detections, ini, ini+200,data_path ,save=True, title='tracking_bgadp_overlap_' + cam)
        gt_video = []
        video_detections = []
        detections_list = []
        gc.collect()


def case3(cam_list):
    """MOG Background Substration + Kalman S03 """
    # S03-C010
    path = 'datasets/aic19-track1-mtmc-train/train/S03/'
    print('Case 3: MOG+Kalman')

    while cam_list:
        cam = cam_list[0]
        cam_list.pop(0)
        if cam == 'c010': ini, fin = 536, 2141
        if cam == 'c011': ini, fin = 571, 2279
        if cam == 'c012': ini, fin = 607, 2422
        if cam == 'c013': ini, fin = 605, 2415
        if cam == 'c014': ini, fin = 584, 2332
        if cam == 'c015': ini, fin = 483, 1928

        data_path = path + cam + '/frames/'
        roi_path = path + cam + '/roi.jpg'
        gt_dir = path + cam + '/gt/gt.txt'
        filename = 'mtsc/detections_bg/detections_MOG_' + cam + '.pkl'
        detections_list = background_subtraction_sota(data_path, roi_path, filename, ini, 'MOG')
        gt_video = Video(Video().getgt(gt_dir, ini, fin))
        [train_list, test_list] = get_frames(data_path, 0.25)
        # visualize_mask2(detections_list, gt_video, ini, fin, test_list, 'bgmog_kalman')
        filename = 'mtsc/detections_bg/case3/results_MOG_kalman_' + cam + '.pkl'
        i = detections_list.list_frames[len(detections_list.list_frames) - 1]
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                video_detections = pickle.load(file)
        else:
            video_detections = tracking_kalman(detections_list, path=path, begin=ini, end=fin, cam=cam, num_color=50,
                                               visualize=False, first_time=False)
            with open(filename, 'wb') as f:
                pickle.dump(video_detections, f)

        compute_IDF1_2(video_detections, gt_video, ini, fin)
        print('S03-' + cam + ' results:')
        print(compute_IDF1_2(video_detections, gt_video, ini, fin))
        gt_video = []
        video_detections = []
        detections_list = []
        gc.collect()


def case4(cam_list):
    """MOG Background Substration + Overlap S03 """
    # S03-C010
    path = 'datasets/aic19-track1-mtmc-train/train/S03/'
    print('Case 4: MOG+Overlap')

    while cam_list:
        cam = cam_list[0]
        cam_list.pop(0)
        if cam == 'c010': ini, fin = 536, 2141
        if cam == 'c011': ini, fin = 571, 2279
        if cam == 'c012': ini, fin = 607, 2422
        if cam == 'c013': ini, fin = 605, 2415
        if cam == 'c014': ini, fin = 584, 2332
        if cam == 'c015': ini, fin = 483, 1928

        data_path = path + cam + '/frames/'
        roi_path = path + cam + '/roi.jpg'
        gt_dir = path + cam + '/gt/gt.txt'
        filename = 'mtsc/detections_bg/detections_MOG_' + cam + '.pkl'
        detections_list = background_subtraction_sota(data_path, roi_path, filename, ini, 'MOG')
        gt_video = Video(Video().getgt(gt_dir, ini, fin))
        filename = 'mtsc/detections_bg/case4/results_MOG_overlap_' + cam + '.pkl'
        i = detections_list.list_frames[len(detections_list.list_frames) - 1]
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                video_detections = pickle.load(file)
        else:
            video_detections = tracking_overlap(detections_list,fin ,path, cam, num_color=50, visualize=False)
            with open(filename, 'wb') as f:
                pickle.dump(video_detections, f)

        print('S03-' + cam + ' results:')
        print(compute_IDF1_overlap(video_detections, gt_video, ini, fin))
        gt_video = []
        video_detections = []
        detections_list = []
        gc.collect()


def case5(cam_list):
    """KNN Background Substration + Kalman S03 """
    # S03-C010
    path = 'datasets/aic19-track1-mtmc-train/train/S03/'
    print('Case 5: KNN+Kalman')

    while cam_list:
        cam = cam_list[0]
        cam_list.pop(0)
        if cam == 'c010': ini, fin = 536, 2141
        if cam == 'c011': ini, fin = 571, 2279
        if cam == 'c012': ini, fin = 607, 2422
        if cam == 'c013': ini, fin = 605, 2415
        if cam == 'c014': ini, fin = 584, 2332
        if cam == 'c015': ini, fin = 483, 1928

        data_path = path + cam + '/frames/'
        roi_path = path + cam + '/roi.jpg'
        gt_dir = path + cam + '/gt/gt.txt'
        filename = 'mtsc/detections_bg/detections_KNN_' + cam + '.pkl'
        detections_list = background_subtraction_sota(data_path, roi_path, filename, ini, 'KNN')
        gt_video = Video(Video().getgt(gt_dir, ini, fin))

        filename = 'mtsc/detections_bg/case5/results_KNN_kalman_' + cam + '.pkl'
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                video_detections = pickle.load(file)
        else:
            video_detections = tracking_kalman(detections_list, path=path, begin=ini, end=fin, cam=cam, num_color=50,
                                               visualize=False, first_time=False)
            with open(filename, 'wb') as f:
                pickle.dump(video_detections, f)


        print('S03-' + cam + ' results:')
        print(compute_IDF1_2(video_detections, gt_video, ini, fin))
        gt_video = []
        video_detections = []
        detections_list = []
        gc.collect()


def case6(cam_list):
    """KNN Background Substration + Overlap S03 """
    # S03-C010
    path = 'datasets/aic19-track1-mtmc-train/train/S03/'
    print('Case 6: KNN+Overlap')

    while cam_list:
        cam = cam_list[0]
        cam_list.pop(0)
        if cam == 'c010': ini, fin = 536, 2141
        if cam == 'c011': ini, fin = 571, 2279
        if cam == 'c012': ini, fin = 607, 2422
        if cam == 'c013': ini, fin = 605, 2415
        if cam == 'c014': ini, fin = 584, 2332
        if cam == 'c015': ini, fin = 483, 1928

        data_path = path + cam + '/frames/'
        roi_path = path + cam + '/roi.jpg'
        gt_dir = path + cam + '/gt/gt.txt'
        filename = 'mtsc/detections_bg/detections_KNN_' + cam + '.pkl'
        detections_list = background_subtraction_sota(data_path, roi_path, filename, ini, 'KNN')
        gt_video = Video(Video().getgt(gt_dir, ini, fin))
        filename = 'mtsc/detections_bg/case6/results_KNN_overlap_' + cam + '.pkl'
        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                video_detections = pickle.load(file)
        else:
            video_detections = tracking_overlap(detections_list,fin ,path, cam, num_color=50, visualize=False)
            with open(filename, 'wb') as f:
                pickle.dump(video_detections, f)

        print('S03-' + cam + ' results:')
        print(compute_IDF1_overlap(video_detections, gt_video, ini, fin))
        gt_video = []
        video_detections = []
        detections_list = []
        gc.collect()


if __name__ == '__main__':
    """Case 1: Adaptive Background Substraction + Kalman
     S03: choose 1 or more camaras(c010,c011,c012,c013,c014,c015)
     """
    cam = ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']
    case1(cam_list=cam)

    """Case 2: Adaptive Background Substraction + Overlap
     S03: choose 1 or more camaras(c010,c011,c012,c013,c014,c015)
     """

    cam = ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']
    case2(cam_list=cam)

    """Case 3: MOG + Kalman
    S03: choose 1 or more camaras(c010,c011,c012,c013,c014,c015)
    """
    cam = ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']
    case3(cam_list=cam)

    """Case 4: MOG + Overlap
    S03: choose 1 or more camaras(c010,c011,c012,c013,c014,c015)
    """

    cam = ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']
    case4(cam_list=cam)

    """Case 5: KNN + Kalman
    S03: choose 1 or more camaras(c010,c011,c012,c013,c014,c015)
    """
    cam = ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']
    case5(cam_list=cam)

    """Case 6: KNN + Overlap
    S03: choose 1 or more camaras(c010,c011,c012,c013,c014,c015)
    """
    cam = ['c010', 'c011', 'c012', 'c013', 'c014', 'c015']
    case6(cam_list=cam)
