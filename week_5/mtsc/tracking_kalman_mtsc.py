from utils.sort import Sort, convert_x_to_bbox
from matplotlib import patches
from PIL import Image
from metrics.evaluation_funcs import *
from utils.tracking_utils import *
import cv2
import matplotlib
matplotlib.use("TkAgg")


def tracking_kalman(detections_list, path, begin, end, cam,num_color,visualize=False, first_time= True):

    gt=path+'/'+ cam +'/gt/gt.txt'
    kalman_tracker = Sort()

    cmap = get_cmap(num_color)

    ax1 = plt.subplot()
    plt.ion()
    video_detections = []

    first_appear = []
    first_apperance=[]

    for frame in detections_list.list_frames:
        detections = []
        frame_num = frame.frame_id
        #print(frame_num)
        for box_detec in frame.bboxes:
            boxes = np.array([box_detec.top_left[0], box_detec.top_left[1], box_detec.width + box_detec.top_left[0],
                              box_detec.height + box_detec.top_left[1]])

            detections.append(boxes)

        trackers = kalman_tracker.update(np.array(detections))

        frame_detections = []

        img_path = path + '/' + cam + '/frames/image' + str(frame_num).zfill(5) + '.jpg'
        #print(img_path)
        hsv = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2HSV)

        im = np.array(Image.open(img_path), dtype=np.uint8)

        for track_state in trackers:
            x, y = track_state[0], track_state[1]
            vx, vy = track_state[-4], track_state[-3]
            id = track_state[-1]
            bbox = convert_x_to_bbox(track_state[:-1])
            bbox = bbox.reshape(bbox.shape[1])

            if bbox[0] < 0: # or bbox[0] > 1920:
                bbox[0] = 0
            # if bbox[2] > 1800:
            #    bbox[2] = 100

            track_det = np.concatenate((bbox, [id])).astype(np.uint64)

            hsv_roi = hsv[track_det[1]:track_det[1] + (track_det[3] - track_det[1]),
                      track_det[0]: track_det[0] + (track_det[2] - track_det[0])]

            histo = np.histogram(hsv_roi, bins=np.linspace(0, 255, 16), density=True)

            frame_detections.append([track_det[4], track_det[0], track_det[1], (track_det[2]-track_det[0]),
                                     (track_det[3] - track_det[1]), histo, frame_num])

            if first_time:
                #if track_det[4] not in [car[1] for car in first_appear]:
                first_appear.append([frame_num, track_det[4], track_det[0], track_det[1], (track_det[2] - track_det[0]),(track_det[3] - track_det[1]), histo])

            color = cmap(int(track_det[4]))
            detec = patches.Rectangle((track_det[0], track_det[1]), (track_det[2]-track_det[0]), (track_det[3]-track_det[1]),
                                      linewidth=1.5, edgecolor=color, facecolor='none')
            if visualize:
                ax1.add_patch(detec)

                length = 10
                if np.abs(vx) + np.abs(vy) > 0.5:
                    ax1.arrow(x, y, vx*length, vy*length, head_width=10, head_length=10, fc=color, ec=color)

                caption_id = "ID: {}".format(track_det[4])
                ax1.text(track_det[0], track_det[1] - 10, caption_id, color=color,fontsize=15)

            if visualize:
                ax1.text(10, 60, "Frame: {}".format(frame_num), color='w', fontsize=13)

        video_detections.append(frame_detections)

        if visualize:
            ax1.axis('off')
            ax1.imshow(im)
            plt.show()
            plt.pause(0.001)
            #plt.savefig("cam_2_images/image_{}".format(frame_num-835),bbox_inches = 'tight')
            plt.waitforbuttonpress(0)
            ax1.clear()

    if first_time:
        return video_detections, first_appear
    else:
        return video_detections

#cam='c004'
#detec_dir = 'dataset/'+cam+'/det/det_mask_rcnn.txt'
#gt = 'dataset/'+cam+'/gt/gt.txt'
#tracks = tracking_kalman(detec_dir, gt, 600, 700, cam, 50, visualize=True, first_time=True)