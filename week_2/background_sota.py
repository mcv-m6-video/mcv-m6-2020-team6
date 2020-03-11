from week_2.model.backgroundSubstration import *
from week_2.utils.preprocessing import *
import cv2
import copy
from PIL import Image
from matplotlib import pyplot as plt


def visualize_mask_detec (gt_video, detections_video, begin, end, images, method):

    num_frames = end-begin

    x = []
    fig = plt.figure()
    ax1 = plt.subplot()
    plt.ion()

    for i in range(num_frames):

        detections_bboxes = detections_video.get_frame_by_id(begin + i)

        gt_bboxes = gt_video.get_frame_by_id(begin + i)

        ax1.imshow(images[(begin-536)+i])

        for bbox in gt_bboxes.bboxes:
            ground = patches.Rectangle(bbox.top_left,
                                       bbox.width, bbox.height,
                                       linewidth=1.75, edgecolor='g', facecolor='none', label='groundtruth')
            ax1.add_patch(ground)

        for bbox_noisy in detections_bboxes.bboxes:
            bb = bbox_noisy.to_result()
            detec = patches.Rectangle(bbox_noisy.top_left,
                                      bbox_noisy.width, bbox_noisy.height,
                                      linewidth=1.5, edgecolor='r', facecolor='none', label='detections')

            ax1.add_patch(detec)

        ax1.axis('off')
        plt.title('Background subtraction using '+method)
        plt.show()
        plt.pause(0.005)
        ax1.clear()


def background_subtraction_sota(method, visualize=False):

    if method == 'MOG':
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=100, nmixtures=2)

    elif method == 'MOG2':
        fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=15)

    elif method == 'KNN':
        fgbg = cv2.createBackgroundSubtractorKNN(100, 550)

    frames_dir = '../datasets/train/S03/c010/frames/'
    roi_path = '../datasets/train/S03/c010/roi.jpg'

    train_list, test_list = get_frames(frames_dir, trainbackground=0.25)

    num_test = len(test_list)
    ini_frame = cv2.imread(train_list[0])

    mask = np.zeros((num_test, ini_frame.shape[0], ini_frame.shape[1]))
    roi = cv2.cvtColor(cv2.imread(roi_path), cv2.COLOR_BGR2GRAY)

    images_morpho = []

    gt_dir = 'D:/Documents/Proves_2/week_2/annotation_fix.txt'

    gt_video = Video(Video().getgroundTruthown(gt_dir, 536, 2141))

    filename = 'detections_sota/detections_sota_' + method + '.pkl'

    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            video_fg = pickle.load(file)
    else:

        for j in range(0, num_test):

            frame = cv2.imread(test_list[j], cv2.IMREAD_GRAYSCALE)
            if method == 'MOG2':
                fgmask = fgbg.apply(frame, 0.01)
            else:
                fgmask = fgbg.apply(frame)

            mask_morpho = morphology_operations(roi*fgmask, kernel_open=(5, 5), kernel_close=(50, 50))

            mask[j, :, :] = copy.deepcopy(mask_morpho)

            images_morpho.append(mask_morpho)

        detections_fg, listofmask, video_fg = connected_component_test(mask, min_area=1500, num_frame=536)

        with open(filename, 'wb') as f:
            pickle.dump(video_fg, f)

        if visualize:
            visualize_mask_detec(gt_video, video_fg, 536, 636, listofmask, method)

    mAP(gt_video, video_fg, True, fname='precision_recall_11_sub_'+method+'.png')

