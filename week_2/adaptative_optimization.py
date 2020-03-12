from week_2.model.backgroundSubstration import *
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import gc


def evaluation_adaptative_non_recursive():

    frames_dir = '../datasets/train/S03/c010/frames/'
    roi_path = '../datasets/train/S03/c010/roi.jpg'

    map_list = []

    rho_values = np.arange(0.0º, 0.1, 0.005)

    for rho in rho_values:

        filename = 'detections_rho_non_recursive/detections_rho_'+str("{:.1f}".format(rho))+'_alpha_4_75.pkl'

        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                video_fg = pickle.load(file)
        else:

            image_list_fg = fg_mask_single_gaussian(frames_dir, roi_path, alpha=4.75, adaptive=True, rho=rho)
            detections_fg, listofmask, video_fg = connected_component_test(image_list_fg, min_area=1500, num_frame=536)

            with open(filename, 'wb') as f:
                pickle.dump(video_fg, f)

        gt_dir = 'annotation_fix.txt'

        gt_video = Video(Video().getgroundTruthown(gt_dir, 536, 2141))

        if video_fg:
            try:
                map = mAP(gt_video, video_fg, False, fname='precision_recall_11_sub_'+str("{:.1f}".format(rho))+'.png')
                map_list.append(map)
            except ValueError:
                map = 0
                map_list.append(map)

        print(rho)

    plt.figure()
    plt.plot(rho_values, map_list)
    plt.title('mAP vs Rho with alpha = 4.75')
    plt.xlabel('rho')
    plt.ylabel('mAP')
    plt.savefig('map_rho_alpha_4_75.png')
    plt.show()


def evaluation_adaptative_recursive():

    frames_dir = '../datasets/train/S03/c010/frames/'
    roi_path = '../datasets/train/S03/c010/roi.jpg'

    map_list = []

    rho_values = np.arange(0.01, 0.31, 0.05)

    alpha_values = np.arange(5.8, 7.4, 0.2)

    Z = np.zeros([len(alpha_values), len(rho_values)])
    i = 0
    j = 0
    best = []
    for alpha in alpha_values:

        for rho in rho_values:

            filename = 'gridsearch_yuv/detections_rho_'+str("{:.1f}".format(rho))+'_alpha_'+str("{:.2f}".format(alpha))+'.pkl'

            if os.path.exists(filename):
                with open(filename, 'rb') as file:
                    video_fg = pickle.load(file)
            else:

                image_list_fg = fg_mask_single_gaussian(frames_dir, roi_path, alpha=alpha, adaptive=True, rho=rho)
                detections_fg, listofmask, video_fg = connected_component_test(image_list_fg, min_area=1500, num_frame=536)
                image_list_fg=[]
                detections_fg=[]
                listofmask=[]
                gc.collect()
                with open(filename, 'wb') as f:
                    pickle.dump(video_fg, f)

            gt_dir = 'annotation_fix.txt'

            gt_video = Video(Video().getgroundTruthown(gt_dir, 536, 2141))

            if video_fg:
                try:
                    map = mAP(gt_video, video_fg, False, fname='precision_recall_11_sub_'+str("{:.1f}".format(rho))+'.png')
                    map_list.append(map)
                except ValueError:
                    map = 0

            Z[i, j] = map

            print(rho)
            print(alpha)

            j = j+1
        i = i+1
        j = 0

    X, Y = np.meshgrid(rho_values, alpha_values)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z,cmap=cm.coolwarm)
    axis = ["Rho", "Alpha", "mAP"]
    ax.set_xlabel(axis[0])
    ax.set_ylabel(axis[1])
    ax.set_zlabel(axis[2])
    fig.colorbar(surf, shrink=0.4, aspect=6)
    plt.title('Alpha and Rho optimization over mAP')
    plt.savefig('grid_search.png')
    plt.show()





