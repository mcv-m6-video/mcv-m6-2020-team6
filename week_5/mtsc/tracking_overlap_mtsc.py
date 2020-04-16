from matplotlib import patches
from PIL import Image
from metrics.evaluation_funcs import *
from collections import Counter
import cv2
from utils.tracking_utils import *
from utils.filtering import *
import matplotlib
matplotlib.use("TkAgg")

def tracking_overlap(detections, end, path,cam,num_color=50, visualize=True):
    tracker = 0
    detec_x_frame = 0
    detections_frame = []
    trackings = []
    old_detections = []
    new_detections = []
    list_detections = []
    ids = []
    video_detections = []
    fig = plt.figure()
    ax1 = plt.subplot()
    plt.ion()
    cmap = get_cmap(num_color)

    for i in range(1, end):

        image_num = i

        img_path = path +'/'+ cam + '/frames/image' + str(image_num).zfill(5) + '.jpg'
        im = np.array(Image.open(img_path), dtype=np.uint8)
        frame = detections.get_frame_by_id(i)

        if len(frame.bboxes)!=0:

            if not trackings:

                detec_x_frame = len(frame.bboxes)
                tracker = tracker + detec_x_frame
                ids = list(range(0, tracker))

                detections_frame = frame.bboxes
            else:
                old_detections = []
                new_detections = []
                for detec_box in frame.bboxes:

                    ious = []

                    #tracks = [[frame, car_id, boxes, histo] for (frame, car_id, boxes, histo) in trackings if frame == image_num-1
                    #          or frame == image_num-2]

                    tracks = [[frame, car_id, boxes] for (frame, car_id, boxes) in trackings if
                              frame == image_num - 1
                              or frame == image_num - 2 or frame == image_num - 3 or frame == image_num - 4 or frame == image_num - 5]

                    for i in trackings:
                        ious.append([i[1], iou_bbox_2(detec_box, i[2]), detec_box, i[0]])
                    if ious:
                        ioubox = max(ious, key=lambda x: (x[1]))

                        if ioubox[1] >= 0.5 and (image_num-ioubox[3]) < 10:
                            old_detections.append([image_num, ioubox[0], ioubox[2]])
                        else:
                            num_new_id = [ide[1] for ide in trackings]

                            try:
                                num_new_id = max(k for k, v in Counter(num_new_id).items() if v>=5)
                            except (ValueError, RuntimeError, TypeError, NameError):
                                num_new_id = max(k for k, v in Counter(num_new_id).items() if v >= 1)

                            new_detections.append([image_num, num_new_id+1+len(new_detections), detec_box])

                old_detections = sorted(old_detections, key=lambda x: x[1])

                list_detections = [row[2] for row in old_detections]

                list_detections.extend([row[2] for row in new_detections])

                ids = [row[1] for row in old_detections]
                ids.extend(row[1] for row in new_detections)

                detections_frame = list_detections

            frame_detections = []



            for detection, id in zip(detections_frame, ids):

                trackings.append([image_num, id, detection])

                frame_detections.append([id, detection.top_left[0], detection.top_left[1], detection.width, detection.height,image_num])

                color = cmap(id)

                box = list([detection.top_left[0], detection.top_left[1], detection.width + detection.top_left[0],
                            detection.height + detection.top_left[1]])

                detec = patches.Rectangle(detection.top_left, detection.width, detection.height, linewidth=1.5, edgecolor=color, facecolor='none')
                ax1.add_patch(detec)

                caption_id = "ID: {}".format(id)
                ax1.text(detection.top_left[0], detection.top_left[1]-10, caption_id, color=color)



            video_detections.append(frame_detections)

        if visualize:
            ax1.text(10, 60, "Frame: {}".format(image_num), color='w', fontsize=11)
            ax1.axis('off')
            ax1.imshow(im)
            plt.show()
            plt.pause(0.001)
            ax1.clear()
    return video_detections

'''
detections = '../detections_dl/S03/mask_rcnn_ft/detections_mask_rcnn_ft_c010_S03.txt'
path_data = '../AIC20_track3/'
split = 'train/'
scene = 'S03'
roi_path = path_data + split + scene + '/' + 'c010' + '/' + 'roi.jpg'
detections_list = Video(Video().getgt_detections(detections, 0, 2000))
detec_list = preprocess_videodetections(detections_list, 2000, roi_path)
tracking_overlap(detec_list, 2000,path_data+split+scene,'c010',num_color=50, visualize=True)
'''