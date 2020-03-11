import numpy as np
import cv2
from week_2.model.bbox import BBox
from week_2.model.video import *
from matplotlib import patches
from week_2.metrics.evaluation_funcs import *
from week_2.model.backgroundSubstration import *
from collections import Counter
from read_annotation import *
import gc

def preprocess_annotations(gt_dir):
    """
    preprocess_annotations(gt_dir)
    The parked cars are erased from the gt, in order to evaluate our background foreground algorithm.
    :param gt_dir:
    :return:
    """
    read_annotations("m6-full_annotation.xml")

    gt_video = Video(Video().getgroundTruthown(gt_dir, 0, 2141))
    frames_dir = '../datasets/train/S03/c010/frames/'
    train_list, test_list = get_frames(frames_dir, trainbackground=0.25)

    listbboxes = gt_video.get_detections_all()
    listoftopleft=([i.top_left for i in listbboxes])

    bboxes_r=list([k for k, v in Counter(map(tuple,listoftopleft)).items() if v > 500])
    bboxes_r=list([list(i) for i in bboxes_r])
    trackids=[]
    [(listbboxes.remove(i),trackids.append(i.track_id)) for i in listbboxes if(i.top_left in bboxes_r)]
    trackids=set(trackids)

    for i in trackids:
        for j in gt_video.get_detections_by_track(i):
            if j in listbboxes:
                listbboxes.remove(j)

    detect_list=[]
    for i in listbboxes:
        detect_list.append(
            str(i.frame_id) + ',' + str(-1) + ',' + str(i.top_left[0]) + ',' + str(i.top_left[1]) + ','
            + str(i.width) + ',' + str(i.height) + ',' + str(i.track_id) + ',' + str(-1) + ',' + str(-1)
            + ',' + str(-1))

    detections = sorted(detect_list, key=lambda x: int(x.split(',')[0]))
    detec_file = open('annotation_fix.txt', 'w')

    for i in detections:
        detec_file.writelines(i + '\n')
    detec_file.close()

    gt_vide2 = Video(Video().getgroundTruthown("annotation_fix.txt", 0, 2141))
    #visualize_mask_2(gt_vide2, 0, 2141, train_list)

def fill_holes(mask: np.array) -> np.array:
    """
    get_mask(mask)

    Fills the holes of closed areas.

    Parameters   Value
   ----------------------
    'mask'       Binary image with the detections obtained by the image segmentation

    returns the improved mask

    """


    im_floodfill = mask.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_floodfill.shape[:2]
    filling_mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(im_floodfill, filling_mask, (0, 0), 255)

    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    im_out = mask | im_floodfill_inv

    return im_out


def morphology_operations(mask: np.array, kernel_open=(5, 5), kernel_close=(50, 50)):
    """
    morphology_operation(mask)

    Function to apply morphological operatations (Opening and Closing) to the resulting masks

    Parameters    Value
   ----------------------
    'mask'        Binary image with the signals detections obtained by the image segmentation


    Returns the modified mask
    """

    kernel_open_mat = np.ones(kernel_open, np.uint8)
    kernel_close_mat = np.ones(kernel_close, np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open_mat)
    mask = fill_holes(mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close_mat)

    return mask


def connected_component(mask, min_area=1200, frameid=1, color="gray"):
    listBBox = []
    connectivity = 12

    if color!="gray":
            if color=="yuv":
                maskx = cv2.bitwise_and(mask[:,:,0], mask[:,:,1])
                masky = cv2.bitwise_and(mask[:,:,0], mask[:,:,2])
                mask = cv2.bitwise_or(maskx[:,:], masky[:,:])
            else:
                mask = mask[:, :, 0]

    output = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity, cv2.CV_32S)

    labels = output[1]
    stats = output[2]

    for i in range(stats.shape[0]):

        if stats[i][4] > min_area:
            width = stats[i][2]
            height = stats[i][3]
            if not (stats[i][0], stats[i][1]) == (0, 0):
                bbox = BBox(frameid, -1 ,(stats[i][0], stats[i][1]), width, height, 1)
                for j in listBBox:
                    if iou_bbox_2(j, bbox) > 0.3:
                        bbox = bbox.union(j)
                        listBBox.remove(j)

                listBBox.append(bbox)

    valid_areas_index = np.where(stats[1:, 4] >= min_area)[0] + 1

    valid_index = np.isin(labels, valid_areas_index)
    mask = mask * valid_index

    return mask, listBBox


def connected_component_test(image_list_fg,min_area=1200, num_frame=536,color="gray"):
    Bboxes_fg = []
    listofmask = []
    list_frames = []
    gc.collect()
    for i in image_list_fg:
        mask, listBBox = connected_component(i, min_area, num_frame,color)

        list_frames.append(Frame(num_frame, listBBox))
        Bboxes_fg = Bboxes_fg + listBBox
        listofmask.append(mask)
        num_frame = num_frame + 1
    video_fg = Video(list_frames)
    return Bboxes_fg, listofmask, video_fg

def connected_component_test2(image_list_fg,testlist,min_area=1200,color="gray"):
    Bboxes_fg = []
    listofmask = []
    list_frames = []
    j=0
    for i in image_list_fg:
        num_frame = int(testlist[j].split('image', 1)[1].split('.', 1)[0])
        mask, listBBox = connected_component(i, min_area, num_frame,color)

        list_frames.append(Frame(num_frame, listBBox))
        Bboxes_fg = Bboxes_fg + listBBox
        listofmask.append(mask)
        j=j+1

    video_fg = Video(list_frames)
    return Bboxes_fg, listofmask, video_fg

def get_video_frameid(gt_video, frameid):
    j = 0
    frames = gt_video[:frameid]

    gt_vid = Video(frames)

    return gt_vid


def show_bboxes(bboxes, frame):
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(frame, cmap='gray')

    i = 0
    for bbox in bboxes:
        rect = patches.Rectangle((bbox.top_left),
                                 bbox.width, bbox.height,
                                 linewidth=3, edgecolor='g', facecolor='none')
        ax.add_patch(rect)

    plt.axis('off')
    plt.show()


def visualize(test_list, listbboxes):
    j = 0
    for i in test_list:
        bboxes_frame = list(filter(lambda obj: obj.frame_id == j, listbboxes))
        frame = cv2.imread(i)
        show_bboxes(bboxes_frame, frame)
        j = j + 1
