import cv2
import numpy as np
from PIL import Image
from matplotlib import patches, pyplot as plt

from histogram_comparison import compare_descriptors
from tracking_kalman import tracking_kalman_color
from model.video import Video
from utils.filtering import preprocess_videodetections
from itertools import combinations
from collections import Counter
import os
import pickle
from utils.iou_over_time_RT import visualize_vtracking


class Camera:
    def __init__(
            self, sequence, camera, time_delay, detect_dir, frame_rate=10, working_set="train", ini_frame=0,
            fin_frame=1995
    ):
        self.sequence = sequence
        self.camera = camera
        self.time_next_frame = time_delay
        self.frame_rate = frame_rate
        self.ini_frame = ini_frame
        self.fin_frame = fin_frame
        self.detect_dir = detect_dir


def mtmc_comparison_features(cam, cam_ref, detect_dir, detect_dir_ref, frame_diff):
    path = 'datasets/aic19-track1-mtmc-train/train/' + cam.sequence + '/' + cam.camera + '/'
    path_ref = 'datasets/aic19-track1-mtmc-train/train/' + cam_ref.sequence + '/' + cam_ref.camera + '/'
    gt, gt_ref = (path + 'gt/gt.txt', path_ref + 'gt/gt.txt')
    roi_path, roi_path_ref = (path + 'roi.jpg', path_ref + 'roi.jpg')


    filename = 'tracks.pkl'
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            tracks,detections_list,video_ref = pickle.load(file)
    else:
        tracks_cam_ref, first_appear_ref = tracking_kalman_color(detect_dir_ref, roi_path_ref, cam_ref.ini_frame,
                                                                 cam_ref.fin_frame, cam_ref.camera, 80,
                                                                 visualize=False, first_time=True)

        video_ref = Video().track2video_kalman(tracks_cam_ref, cam_ref.ini_frame, cam_ref.fin_frame)

        tracks_cam, first_appear = tracking_kalman_color(detect_dir, roi_path, cam.ini_frame,
                                                         cam.fin_frame, cam.camera, 80,
                                                         visualize=False, first_time=True)
        detections_list = Video().track2video_kalman(tracks_cam, cam.ini_frame, cam.fin_frame)


        method = 'SIFT'
        if method == 'ORB':
            finder = cv2.ORB_create()
        elif method == 'SIFT':
            finder = cv2.xfeatures2d.SIFT_create()
        elif method == 'SURF':
            finder = cv2.xfeatures2d.SURF_create()

        min_score = 0.1
        tracks = []
        filename = 'tracks.pkl'
        for frame in detections_list.list_frames:
            frame_num = frame.frame_id
            print(frame_num)

            image_path = path + '/frames/image' + str(frame_num).zfill(5) + '.jpg'

            img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)

            im = np.array(Image.open(image_path), dtype=np.uint8)

            for detec_box in frame.bboxes:
                best_val = []

                img_roi = img[int(detec_box.top_left[1] + detec_box.height * 0.3):int(
                    detec_box.top_left[1] + detec_box.height * 0.7),
                          int(detec_box.top_left[0] + detec_box.width * 0.3): int(
                              detec_box.top_left[0] + detec_box.width * 0.7)]

                kp1, des1 = finder.detectAndCompute(img_roi, None)

                for car in first_appear_ref:

                    if frame_num - frame_diff == car[0]:
                        image_path_ref = path_ref + '/frames/image' + str(car[0]).zfill(5) + '.jpg'
                        img_ref = cv2.cvtColor(cv2.imread(image_path_ref), cv2.COLOR_BGR2GRAY)

                        img_roi_ref = img_ref[int(car[3] + car[5] * 0.3):int(car[3] + car[5] * 0.7),
                                      int(car[2] + car[4] * 0.3): int(car[2] + car[4] * 0.7)]

                        kp2, des2 = finder.detectAndCompute(img_roi_ref, None)
                        if kp2:
                            val_features = compare_descriptors(des1, kp2, des2)
                            best_val.append([val_features, car[1], detec_box.det_id])
                    if best_val:
                        best_candidate = max(best_val, key=lambda t: t[0])
                        # print(best_candidate[0])
                        if best_candidate[0] > min_score:
                            print(best_candidate[0])
                            tracks.append(
                                [best_candidate[0], best_candidate[1], best_candidate[2]])
            with open(filename, 'wb') as f:
                pickle.dump([tracks,detections_list,video_ref], f)

    num_tracks = detections_list.get_num_tracks()
    corr = []
    tracksids = []
    for j in num_tracks:
        track_select = [i for i in tracks if i[2] == j]
        if track_select:
            tracksids = max(track_select, key=lambda t: t[0])

        if tracksids:
            corr.append([tracksids[1], j])
    new=detections_list.change_track(corr, cam.fin_frame)
    gt_video = Video(Video().getgt_detections(gt, cam.ini_frame, cam.fin_frame))
    gt_video_ref = Video(Video().getgt_detections(gt_ref, cam_ref.ini_frame, cam_ref.fin_frame))

    visualize_vtracking(
            video_ref, gt_video_ref, cam_ref.ini_frame, cam_ref.fin_frame, path_ref+'frames/', save=True, title="tracking_c2"
    )

    visualize_vtracking(
        new, gt_video, cam.ini_frame, cam.fin_frame, path + 'frames/', save=True, title="tracking_x3"
    )


cam1 = Camera(sequence='S01', camera='c001', time_delay=4, frame_rate=10, working_set="train", ini_frame=1,
              fin_frame=1955, detect_dir='detections_dl/s01/mask_rcnn_ft/detections_mask_rcnn_ft_c001_S01.txt')
cam2 = Camera(sequence='S01', camera='c002', time_delay=4, frame_rate=10, working_set="train", ini_frame=840,
              fin_frame=1000, detect_dir='detections_dl/s01/mask_rcnn_ft/detections_mask_rcnn_ft_c002_S01.txt')
cam3 = Camera(sequence='S01', camera='c003', time_delay=4, frame_rate=10, working_set="train", ini_frame=836,
              fin_frame=1000, detect_dir='detections_dl/s01/mask_rcnn_ft/detections_mask_rcnn_ft_c003_S01.txt')
cam4 = Camera(sequence='S01', camera='c004', time_delay=4, frame_rate=10, working_set="train", ini_frame=1,
              fin_frame=2110, detect_dir='detections_dl/s01/mask_rcnn_ft/detections_mask_rcnn_ft_c004_S01.txt')
cam5 = Camera(sequence='S01', camera='c005', time_delay=4, frame_rate=10, working_set="train", ini_frame=1,
              fin_frame=2110, detect_dir='detections_dl/s01/mask_rcnn_ft/detections_mask_rcnn_ft_c005_S01.txt')

camara_combinations = list(combinations([cam1, cam2, cam3, cam4, cam5], 2))
# Sequence S01
"""for cam,cam_ref in camara_combinations:
    mtmc_comparison_features_tr(cam=cam, cam_ref=cam_ref, detect_dir=cam.detect_dir, detect_dir_ref=cam_ref.detect_dir)"""

mtmc_comparison_features(cam=cam3, cam_ref=cam2, detect_dir=cam3.detect_dir, detect_dir_ref=cam2.detect_dir,
                         frame_diff=4)
