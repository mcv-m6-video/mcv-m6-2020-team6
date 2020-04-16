import os
import pickle

from metrics.evaluation_funcs import compute_IDF1_2, compute_IDF1_voverlap
from model.video import Video
from mtsc.tracking_kalman_mtsc import tracking_kalman
from mtsc.tracking_overlap_mtsc import tracking_overlap
from utils.filtering import preprocess_videodetections


def cam_info(directory):
    info_list = {}
    txt_gt = open(directory, "r")

    for line in txt_gt:
        splitLine = line.split("\n")
        info = splitLine[0].split(" ")[1]
        camera = splitLine[0].split(" ")[0]
        info_cam = float(info)
        info_list[camera] = info_cam

    return info_list


def mstc_nn(cam_list, seq, model, tracking):
    print(tracking + ' tracking using ' + model + ':')
    print('\n')
    path_detections = 'detections_dl'
    path_data = 'AIC20_track3/'
    split = 'train/'

    frame_num_dir = path_data + 'cam_framenum/' + seq + '.txt'
    framenum = cam_info(frame_num_dir)

    if cam_list == 'all':
        cameras = [f.path.split('/')[3] for f in os.scandir(path_data + split + seq + '/') if f.is_dir()]
    else:
        cameras = cam_list

    for cam in cameras:

        gt_dir = path_data + split + seq + '/' + cam + '/' + 'gt/gt.txt'
        detect_path = path_detections + '/' + seq + '/' + model + '/detections_' + model + '_' + cam + '_' + seq + '.txt'
        roi_path = path_data + split + seq + '/' + cam + '/' + 'roi.jpg'
        detections_list = Video(Video().getgt_detections(detect_path, 0, int(framenum[cam])))
        gt_video = Video(Video().getgt(gt_dir, 0, int(framenum[cam])))
        detec_list = preprocess_videodetections(detections_list, int(framenum[cam]), roi_path)
        filename = 'mtsc/detections_nn/' + seq + '/' + 'detections_' + model + '_' + cam + '_' + tracking + '_' + seq + '.pkl'

        if tracking == 'Kalman':

            if os.path.exists(filename):
                with open(filename, 'rb') as file:
                    video_detections = pickle.load(file)
            else:
                video_detections = tracking_kalman(detec_list, path_data + split + seq, 0, int(framenum[cam]), cam, 50,
                                                   visualize=False, first_time=False)
                with open(filename, 'wb') as f:
                    pickle.dump(video_detections, f)

            print(seq + '-' + cam + ' results:')
            compute_IDF1_2(video_detections, gt_video, 0, int(framenum[cam]))
            print('\n')

        elif tracking == 'Overlap':

            if os.path.exists(filename):
                with open(filename, 'rb') as file:
                    video_detections = pickle.load(file)
            else:
                video_detections = tracking_overlap(detec_list, int(framenum[cam]), path_data + split + seq, cam,
                                                    num_color=150, visualize=False)
                with open(filename, 'wb') as f:
                    pickle.dump(video_detections, f)

            track_video = Video().track2video_overlap(video_detections, 0, int(framenum[cam]))

            print(seq + '-' + cam + ' results:')
            compute_IDF1_voverlap(track_video, gt_video, 0, int(framenum[cam]))
            print('\n')


if __name__ == '__main__':
    '''Multi-tracking single camera:
    
    
     cam_list and seq options:
     
     - S01: choose some cameras between ['c001','c002','c003','c004','c005'] or choose 'all'
     - S03: choose some cameras between ['c010','c011','c012',c013','c014','c015'] or choose 'all'
     - S04: choose some cameras between ['c016','c017','c018',c019','c020','c021','c022','c023','c024',c025','c026,'c027','c028','c029','c030',
     'c031','c032,'c033','c034','c035','c036','c037','c038','c039','c040'] or choose 'all'
     
     model options:
      
     - mask
     - mask_rcnn_ft 
     - retinanet
     - yolo
     
     tracking options:
     
     - Kalman
     - Overlap

     '''

    cam = 'all'
    seq = 'S03'
    mstc_nn(cam_list=cam, seq=seq, model='retinanet', tracking='Overlap')

    cam = 'all'
    seq = 'S03'
    mstc_nn(cam_list=cam, seq=seq, model='yolo', tracking='Overlap')
