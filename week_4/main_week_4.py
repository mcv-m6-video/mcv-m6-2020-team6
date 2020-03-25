from week_4.optical_flow_off_the_shelf import *
from week_4.metrics.Optical_flow_metrics import *
from week_4.metrics.Optical_flow_visualization import *
from week_4.metrics.blockmatching import *
from week_4.videostabilization_off_the_shelf import *
from week_4.Tracking_OF_Kalman import *
import cv2


def task11(grid_search,seq,visualization):
    """Optical Flow with Block Matching"""

    if seq == 45:

        gt_dir = "../datasets/kitti/groundtruth/000045_10.png"
        img1 = '../datasets/kitti/images/000045_10.png'
        img2 = '../datasets/kitti/images/000045_11.png'
        test_dir = '../datasets/kitti/results/LKflow_000045_10.png'
    if seq == 157:

        gt_dir = "../datasets/kitti/groundtruth/000157_10.png"
        img1 = '../datasets/kitti/images/000157_10.png'
        img2 = '../datasets/kitti/images/000157_11.png'
        test_dir = '../datasets/kitti/results/LKflow_000157_10.png'

    if not grid_search:

        flow=blockMatching(img1,img2,blocksize=26,searcharea=58,compensation="Backward", method=cv2.TM_CCOEFF_NORMED)
        flow_gt, flow_test = flow_read(gt_dir, test_dir)
        MSEN = msen(flow_gt, flow, seq, 'blockmatching')
        PEPN = pepn(flow_gt, flow, 3)
        print(MSEN)
        print(PEPN)

        if visualization:
            OF_quiver_visualization_flow2d(img1, flow, step=15,
                                         fname_output='flow_' + str(seq) + '_blockMatching_quiver.png')



    if grid_search:


        blocksizes = np.arange(8, 30, 2)
        searchareas = np.arange(30, 92, 4)

        grid_searchBlockMatching(blocksizes,searchareas,img1,img2,gt_dir,test_dir,seq,compensation='Forward')
        grid_searchBlockMatching(blocksizes,searchareas,img1,img2,gt_dir,test_dir,seq,compensation='Backward')




def task12(seq, method, option):
    """Off-the-shelf Optical Flow"""

    if seq == 45:

        gt_dir = "../datasets/kitti/groundtruth/000045_10.png"
        img1 = '../datasets/kitti/images/000045_10.png'
        img2 = '../datasets/kitti/images/000045_11.png'
        test_dir = '../datasets/kitti/results/LKflow_000045_10.png'
    if seq == 157:

        gt_dir = "../datasets/kitti/groundtruth/000157_10.png"
        img1 = '../datasets/kitti/images/000157_10.png'
        img2 = '../datasets/kitti/images/000157_11.png'
        test_dir = '../datasets/kitti/results/LKflow_000157_10.png'

    if method == 'pyflow':

        flow, flow_1 = Coarse2Fine(img1, img2, visualize=True)
        flow_gt, flow_test = flow_read(gt_dir, test_dir)
        MSEN = msen(flow_gt, flow, seq, method)
        PEPN = pepn(flow_gt, flow, 3)
        print(MSEN)
        print(PEPN)

    elif method == 'farneback':

        flow, flow_1 = farneback(img1, img2, visualize=True)
        flow_gt, flow_test = flow_read(gt_dir, test_dir)
        MSEN = msen(flow_gt, flow, seq, method)
        PEPN = pepn(flow_gt, flow, 3)

        print(MSEN)
        print(PEPN)

    elif method == 'horn':

        flow, flow_1 = horn_schunck(img1, img2, visualize=True)
        flow_gt, flow_test = flow_read(gt_dir,test_dir)
        MSEN = msen(flow_gt, flow, seq, method)
        PEPN = pepn(flow_gt, flow, 3)

        print(MSEN)
        print(PEPN)

    elif method == 'lucas_kanade':

        flow, flow_1 = Lucas_kanade(img1, img2, visualize=True)
        flow_gt, flow_test = flow_read(gt_dir,test_dir)
        MSEN = msen(flow_gt, flow, seq, method)
        PEPN = pepn(flow_gt, flow, 3)

        print(MSEN)
        print(PEPN)

    if option == 'color_based':
        OF_visualization_flow(flow, fname_output='flow_'+str(seq)+'_'+method+'_color.png', maxflow=None)
    elif option == 'quiver_based':
        OF_quiver_visualization_flow(img1, flow_1, step=15, fname_output='flow_'+str(seq)+'_'+method+'_quiver.png')


def task21():
    """Video stabilization with Block Matching"""


def task22(method):
    """Off-the-shelf Stabilization
    method 1: Video Stabilization using one type of descriptor ORB and FAST: http://nghiaho.com/?p=2093
    method 2: Video Stabilization using L1 Optimal Camera Paths: https://ieeexplore.ieee.org/document/5995525 "
    """

    videoStabilization_off(method)


def task31():
    """Object Tracking with Optical Flow"""
    detec_dir = 'detections_mask_rcnn_fine_tune.txt'

    gt = 'annotation_only_cars.txt'

    tracking_kalman_of(detec_dir, gt, 1, 2141, 50, visualize=True)



if __name__ == '__main__':
    task11(grid_search=True,seq=45,visualization=True)
    #task12(45, 'farneback', 'quiver_based')
    task21()
    task22(method=1)
    task31()
