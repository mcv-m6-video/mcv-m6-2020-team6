from tracking_kalman import tracking_kalman_embedding
from matplotlib import patches
from PIL import Image
from metrics.evaluation_funcs import *
import cv2
from model.video import *
from utils.tracking_utils import *
from EmbeddingNet.embedding_net.model_emb import EmbeddingNet
from EmbeddingNet.embedding_net.utils_emb import parse_net_params
from operator import itemgetter
from itertools import groupby
from utils.filtering import preprocess_videodetections
import os

def calculate_distances(encodings, car_encoding):
    return np.sqrt(np.sum((car_encoding - np.array(encodings)) ** 2, axis=1))


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


path_detections = 'detections_dl'
path_data = 'AIC20_track3/'
split = 'train/'
scene = 'S01'
method = 'mask_rcnn_ft'
cam_ref = 'c004'
#cam = 'c004'
fps = 10

visualize = True
save_plot = True
color_num = 300

frame_num_dir = path_data + 'cam_framenum/' + scene + '.txt'
timestamp_dir = path_data + 'cam_timestamp/' + scene + '.txt'
framenum = cam_info(frame_num_dir)
timestamp = cam_info(timestamp_dir)

config_path = 'ai_city_challenge_resnext50.yml'
weights_path = 'EmbeddingNet/work_dirs/ai_city_challenge/weights/best_model_resnet50_ai_city.h5'
encodings_path = 'EmbeddingNet/work_dirs/ai_city_challenge/encodings/encodings_resnet50_ai_city.pkl '
config = 'EmbeddingNet/configs/ai_city_challenge_resnext50.yml'

cfg_params = parse_net_params(config)
model_embedding = EmbeddingNet(cfg_params, training=False)
model_embedding.load_model(weights_path)
model_embedding.load_encodings(encodings_path)

detec_dir = path_detections+'/'+scene+'/'+method+'/detections_'+method+'_'+cam_ref+'_'+scene+'.txt '
roi_ref = path_data+split+scene+'/'+cam_ref+'/'+'roi.jpg'
gt = path_data+split+scene+'/'+cam_ref+'/'+'gt/gt.txt'

cameras = [f.path.split('/')[3] for f in os.scandir(path_data+split+scene+'/') if f.is_dir()]

cameras.remove(cam_ref)

for cam in cameras:

    frame_diff = abs(round((timestamp[cam_ref] - timestamp[cam]) * fps))
    tracks_cam, apperance = tracking_kalman_embedding(detec_dir, gt, frame_diff, int(framenum[cam_ref]), cam_ref,
                                                      path_data + split + scene, roi_ref, color_num, model_embedding,
                                                      visualize=True, save=True, first_time=True)

    detec_dir_1 = path_detections+'/'+scene+'/'+method+'/detections_'+method+'_'+cam+'_'+scene+'.txt'
    gt_1 = path_data+split+scene+'/'+cam+'/'+'gt/gt.txt'
    roi_path = path_data+split+scene+'/'+cam+'/'+'roi.jpg'
    gt_list = Video(Video().getgt(gt_1, 0, int(framenum[cam])))
    detections_list = Video(Video().getgt_detections(detec_dir_1, 0, int(framenum[cam])))
    detections_list = preprocess_videodetections(detections_list, int(framenum[cam]), roi_path)

    ax1 = plt.subplot()
    plt.ion()
    video_detections = []

    detections_cam = []
    cmap = get_cmap(color_num)
    tracks = []

    for frame in detections_list.list_frames:
        detections = []
        frame_num = frame.frame_id
        print(frame_num)

        path = path_data+split+scene+'/'+cam+'/'+'frames/image'+str(frame_num).zfill(5)+'.jpg'

        im = np.array(Image.open(path), dtype=np.uint8)
        imm = cv2.imread(path)

        frame_detections = []
        frame_detec = []

        for detec_box in frame.bboxes:

            best_val = []
            best_candidate = []

            image = imm[int(detec_box.top_left[1]):int(detec_box.top_left[1] + detec_box.height),
                    int(detec_box.top_left[0]): int(detec_box.top_left[0] + detec_box.width)]

            encoding = model_embedding.predict_encoding(image)

            same_frame = [x for x in apperance if frame_num - frame_diff == x[0]]

            for car in same_frame:
                encoding_car = car[6]

                similarity = calculate_distances(encoding, encoding_car)

                best_val.append([similarity, car[1], detec_box, image])

            if best_val and best_val[0]:

                best_candidate = min(best_val, key=lambda t: t[0])

                if best_candidate[0] <= 0.92:

                    if len(tracks) != 0:

                        last_frame = [[frame, car_id, top_left_0, top_left_1, width, height] for
                                      (frame, car_id, top_left_0, top_left_1, width, height, img) in tracks if
                                      frame == frame_num - 1]

                        current_id = [[frame, car_id, top_left_0, top_left_1, width, height] for
                                      (frame, car_id, top_left_0, top_left_1, width, height) in last_frame if
                                      car_id == best_candidate[1]]

                        if len(current_id) != 0:

                            if abs(current_id[0][2] - best_candidate[2].top_left[0]) <= 100 and abs(
                                    current_id[0][3] - best_candidate[2].top_left[1]) <= 100:
                                frame_detections.append(best_candidate)
                        else:
                            frame_detections.append(best_candidate)
                    else:
                        frame_detections.append(best_candidate)

        if frame_detections:

            frame_detections.sort(key=itemgetter(1))
            result = []

            for _, g in groupby(frame_detections, itemgetter(1)):
                result.append(min(g, key=itemgetter(0)))

            for final_tracks in result:
                tracks.append([frame_num, final_tracks[1], final_tracks[2].top_left[0], final_tracks[2].top_left[1],
                               final_tracks[2].width, final_tracks[2].height, final_tracks[3]])
                frame_detec.append(
                    [final_tracks[1], final_tracks[2].top_left[0], final_tracks[2].top_left[1], final_tracks[2].width,
                     final_tracks[2].height, final_tracks[3], frame_num])

                color = cmap(int(final_tracks[1]))

                detec = patches.Rectangle(final_tracks[2].top_left, final_tracks[2].width, final_tracks[2].height,
                                          linewidth=1.5, edgecolor=color,
                                          facecolor='none')
                ax1.add_patch(detec)

                caption_id = "ID: {}".format(str(final_tracks[1]))
                ax1.text(final_tracks[2].top_left[0], final_tracks[2].top_left[1] - 10, caption_id, color=color,
                         fontsize=15)

            video_detections.append(frame_detec)

        if visualize:

            ax1.text(10, 60, "Frame: {}".format(frame_num), color='w', fontsize=13)
            ax1.axis('off')
            ax1.imshow(im)
            plt.show()
            plt.pause(0.001)
            if save_plot:
                plt.savefig('images/'+scene+'/'+cam+'/image_{}'.format(frame_num), bbox_inches='tight')
            ax1.clear()

    print('\n')
    print(cam)
    compute_IDF1_2(video_detections, gt_list, frame_diff, int(framenum[cam]))
    print('\n')