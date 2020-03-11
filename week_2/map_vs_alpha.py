from week_2.model.backgroundSubstration import *
import numpy as np


def evaluation_map_vs_alpha():

    frames_dir = '../datasets/train/S03/c010/frames/'
    roi_path = '../datasets/train/S03/c010/roi.jpg'

    train_list, test_list = get_frames(frames_dir, trainbackground=0.25)

    map_list = []

    alpha_values = np.arange(0.75, 12.25, 0.25)

    for alpha in alpha_values:

        filename = 'detections_alpha/detections_alpha_'+str(alpha)+'.pkl'

        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                video_fg = pickle.load(file)
        else:

            image_list_fg = fg_mask_single_gaussian(frames_dir, roi_path, alpha)
            detections_fg, listofmask, video_fg = connected_component_test(image_list_fg, min_area=1500)

            with open(filename, 'wb') as f:
                pickle.dump(video_fg, f)

        gt_dir = 'annotation_fix.txt'

        gt_video = Video(Video().getgroundTruthown(gt_dir, 536, 2141))

        if not video_fg:
            map = 0
        if video_fg:
            map = mAP(gt_video, video_fg, False, fname='precision_recall_11_sub_'+str(alpha)+'.png')
            map_list.append(map)

        print(alpha)

    plt.figure()
    plt.plot(alpha_values, map_list)
    plt.title('mAP vs Alpha')
    plt.xlabel('alpha')
    plt.ylabel('mAP')
    plt.savefig('map_alpha.png')
    plt.show()

def evaluation_map_vs_alpha_color(color):

    frames_dir = '../datasets/train/S03/c010/frames/'
    roi_path = '../datasets/train/S03/c010/roi.jpg'

    train_list, test_list = get_frames(frames_dir, trainbackground=0.25)

    map_list = []

    alpha_values = np.arange(1.5, 8, 0.25)

    for alpha in alpha_values:

        filename = 'detections_alpha_'+color+'/detections_alpha_'+str(alpha)+'.pkl'

        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                video_fg = pickle.load(file)
        else:

            image_list_fg = fg_mask_single_gaussian(frames_dir, roi_path, alpha,colorspace=color)
            detections_fg, listofmask, video_fg = connected_component_test(image_list_fg, min_area=1500,color=color)
            image_list_fg = []
            detections_fg = []
            listofmask = []
            gc.collect()
            with open(filename, 'wb') as f:
                pickle.dump(video_fg, f)

        gt_dir = 'annotation_fix.txt'

        gt_video = Video(Video().getgroundTruthown(gt_dir, 536, 2141))

        if not video_fg:
            map = 0
        if video_fg:
            map = mAP(gt_video, video_fg, False, fname='precision_recall_11_sub_'+color+'_'+str(alpha)+'.png')
            map_list.append(map)

        print(alpha)

    plt.figure()
    plt.plot(alpha_values, map_list)
    plt.title('mAP vs Alpha')
    plt.xlabel('alpha')
    plt.ylabel('mAP')
    figName='map_alpha_'+color+'.png'
    plt.savefig(figName)
    plt.show()






