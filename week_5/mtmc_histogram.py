import os
import pickle
from collections import Counter

import cv2
import numpy as np
from PIL import Image

from histogram_comparison import compare_histograms
from metrics.evaluation_funcs import compute_IDF1_voverlap
from model.video import Video
from tracking_kalman import tracking_kalman_color
from utils.iou_over_time_RT import visualize_vtracking
from utils.tracking_utils import get_cmap


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


def mtmc_comparison(cam, cam_ref, detect_dir, detect_dir_ref, frame_diff, colorspace="rgb", dim=3,
                    method='chi'):
    convert = {
        "gray": cv2.COLOR_BGR2GRAY,
        "hsv": cv2.COLOR_BGR2HSV,
        "rgb": cv2.COLOR_BGR2RGB,
        "yuv": cv2.COLOR_BGR2YUV,
        "lab": cv2.COLOR_BGR2LAB
    }
    dim_h = {
        "hsv": [[16, 16], [0, 180, 0, 255], [16, 16, 16], [0, 180, 0, 255, 0, 255]],
        "rgb": [[32, 16], [0, 255, 0, 255], [16, 16, 16], [0, 255, 0, 255, 0, 255]],
        "yuv": [[16, 16], [0, 255, 0, 255], [16, 16, 16], [0, 255, 0, 255, 0, 255]],
        "lab": [[16, 16], [0, 255, 0, 255], [16, 16, 16], [0, 255, 0, 255, 0, 255]]
    }
    methods = {
        "corr": [cv2.HISTCMP_CORREL, max, 0.5],
        "chi": [cv2.HISTCMP_CHISQR, min, 15],
        "intersection": [cv2.HISTCMP_INTERSECT, max, 0.8],
        "bhatta": [cv2.HISTCMP_BHATTACHARYYA, min, 0.55]
    }

    filename = 'tracks_' + cam.camera + '_' + cam_ref.camera + '.pkl'
    filename2 = 'tracks_' + cam.camera + '_' + cam_ref.camera + '_detectionlist.pkl'
    path = 'datasets/aic19-track1-mtmc-train/train/' + cam.sequence + '/' + cam.camera + '/'
    path_ref = 'datasets/aic19-track1-mtmc-train/train/' + cam_ref.sequence + '/' + cam_ref.camera + '/'
    gt, gt_ref = (path + 'gt/gt.txt', path_ref + 'gt/gt.txt')
    roi_path, roi_path_ref = (path + 'roi.jpg', path_ref + 'roi.jpg')

    if os.path.exists(filename) and os.path.exists(filename2):
        with open(filename, 'rb') as file:
            tracks = pickle.load(file)
        with open(filename2, 'rb') as file:
            detections_list = pickle.load(file)
    else:
        filename_track='trackingKalman'+cam_ref.camera+'.pkl'
        if os.path.exists(filename_track):
            with open(filename_track, 'rb') as file:
                tracks_cam3, first_appear_3 = pickle.load(file)
        else:
            tracks_cam3, first_appear_3 = tracking_kalman_color(detect_dir_ref, roi_path_ref, cam_ref.ini_frame,
                                                            cam_ref.fin_frame, cam_ref.camera, 80,
                                                            visualize=False, first_time=True,colorspace=colorspace, dim=dim)

            with open(filename_track, 'wb') as f:
                pickle.dump([tracks_cam3, first_appear_3], f)

        filename_track = 'trackingKalman' + cam.camera + '.pkl'
        if os.path.exists(filename_track):
            with open(filename_track, 'rb') as file:
                tracks_cam, first_appear  = pickle.load(file)
        else:
            tracks_cam, first_appear = tracking_kalman_color(detect_dir, roi_path, cam.ini_frame,
                                                         cam.fin_frame, cam.camera, 80,
                                                         visualize=False, first_time=True)
            with open(filename_track, 'wb') as f:
                pickle.dump([tracks_cam, first_appear], f)


        detections_list = Video().track2video_kalman(tracks_cam, cam.ini_frame, cam.fin_frame)
        video_tracking_cam_ref = Video().track2video_kalman(tracks_cam3, cam_ref.ini_frame, cam_ref.fin_frame)

        tracks = []

        for frame in detections_list.list_frames:
            frame_num = frame.frame_id
            print(frame_num)

            image_path = path + 'frames/image' + str(frame_num).zfill(5) + '.jpg'

            img = cv2.cvtColor(cv2.imread(image_path), convert[colorspace])


            for detec_box in frame.bboxes:
                best_val = []

                if dim == 3:
                    img_roi = img[int(detec_box.top_left[1]+detec_box.height*0.15):int(detec_box.top_left[1] + detec_box.height*0.75),
                              int(detec_box.top_left[0]+detec_box.width*0.15): int(detec_box.top_left[0] + detec_box.width*0.75), :]
                    histo = cv2.calcHist([img_roi], [0, 1, 2], None, dim_h[colorspace][2], dim_h[colorspace][3])
                else:
                    img_roi = img[int(detec_box.top_left[1]+detec_box.height*0.2):int(detec_box.top_left[1] + detec_box.height*0.8),
                              int(detec_box.top_left[0]+detec_box.width*0.2): int(detec_box.top_left[0] + detec_box.width*0.8)]
                    histo = cv2.calcHist([img_roi], [0, 1], None, dim_h[colorspace][0], dim_h[colorspace][1])

                cv2.normalize(histo, histo, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

                for car in first_appear_3:
                    if frame_num - frame_diff == car[0]:
                        image_path_ref = path_ref + 'frames/image' + str(car[0]).zfill(5) + '.jpg'
                        img_ref= cv2.cvtColor(cv2.imread(image_path_ref), convert[colorspace])
                        if dim == 3:
                            img_roi_ref = img_ref[int(car[3]+car[5]*0.15):int(car[3] + (car[5])*0.75),
                                      int(car[2]++car[4]*0.15):int(car[2] + (car[4])*0.75), :]
                            histo2 = cv2.calcHist([img_roi_ref], [0, 1, 2], None, dim_h[colorspace][2], dim_h[colorspace][3])
                        else:
                            img_roi_ref = img_ref[int(car[2]+car[4]*0.2):int(car[2] + (car[4])*0.8),
                                      int(car[3]+car[5]*0.2):int(car[3] + (car[5])*0.8), :]
                            histo2 = cv2.calcHist([img_roi_ref], [0, 1], None, dim_h[colorspace][0], dim_h[colorspace][1])


                        mi=sum(sum(sum(histo2)))
                        cv2.normalize(histo2, histo2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                        val_hist = compare_histograms(histo,histo2, methods[method][0])

                        best_val.append([val_hist, car[1], detec_box.det_id])

                    if best_val:
                        best_candidate = methods[method][1](best_val, key=lambda t: t[0])
                        print(best_candidate)
                        if methods[method][1]==max:
                            if best_candidate[0] > methods[method][2]:
                                tracks.append(
                                    [best_candidate[0], best_candidate[1], best_candidate[2]])
                        else:
                            if best_candidate[0] < methods[method][2]:
                                print(best_candidate[0])
                                tracks.append(
                                    [best_candidate[0], best_candidate[1], best_candidate[2]])
        """
        with open(filename, 'wb') as f:
            pickle.dump(tracks, f)
        with open(filename2, 'wb') as f:
            pickle.dump(detections_list, f)
            
        """

    num_tracks = detections_list.get_num_tracks()
    corr = []
    tracksids = []
    for j in num_tracks:
        track_select=[i for i in tracks if i[2]==j]
        if track_select:
            tracksids = methods[method][1](track_select, key= lambda t: t[0])

        if tracksids:
            corr.append([tracksids[1],j])


    new=detections_list.change_track(corr,cam.fin_frame)
    gt_video = Video(Video().getgt_detections(gt, cam.ini_frame, cam.fin_frame))

    gt_video_ref=Video(Video().getgt_detections(gt_ref, cam_ref.ini_frame, cam_ref.fin_frame))

    visualize_vtracking(
        new, gt_video, cam.ini_frame, cam.fin_frame, path + 'frames/', save=True, title="tracking_cam3"
        )
    visualize_vtracking(
        video_tracking_cam_ref, gt_video_ref, cam_ref.ini_frame, cam_ref.fin_frame, path_ref + 'frames/', save=True, title="tracking_cam2"
        )





cam1 = Camera(sequence='S01', camera='c001', time_delay=4, frame_rate=10, working_set="train", ini_frame=1,
fin_frame = 1955, detect_dir = 'detections_dl/s01/mask_rcnn_ft/detections_mask_rcnn_ft_c001_S01.txt')
cam2 = Camera(sequence='S01', camera='c002', time_delay=4, frame_rate=10, working_set="train", ini_frame=840,
fin_frame = 1500, detect_dir = 'detections_dl/s01/mask_rcnn_ft/detections_mask_rcnn_ft_c002_S01.txt')
cam3 = Camera(sequence='S01', camera='c003', time_delay=4, frame_rate=10, working_set="train", ini_frame=836,
fin_frame = 1500, detect_dir = 'detections_dl/s01/mask_rcnn_ft/detections_mask_rcnn_ft_c003_S01.txt')
cam4 = Camera(sequence='S01', camera='c004', time_delay=4, frame_rate=10, working_set="train", ini_frame=1,
fin_frame = 2110, detect_dir = 'detections_dl/s01/mask_rcnn_ft/detections_mask_rcnn_ft_c004_S01.txt')
cam5 = Camera(sequence='S01', camera='c005', time_delay=4, frame_rate=10, working_set="train", ini_frame=1,
fin_frame = 2110, detect_dir = 'detections_dl/s01/mask_rcnn_ft/detections_mask_rcnn_ft_c005_S01.txt')

mtmc_comparison(cam=cam3, cam_ref=cam2, detect_dir=cam3.detect_dir, detect_dir_ref=cam2.detect_dir, frame_diff=4)
