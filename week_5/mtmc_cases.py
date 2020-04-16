import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from PIL import Image

from homography import read_homography
from metrics.evaluation_funcs import compute_IDF1
from model.video import Video
from utils.sort import Sort, convert_x_to_bbox


class Camera:
    def __init__(
        self, sequence, camera, time_delay, frame_rate=10, working_set="train"
    ):
        self.sequence = sequence
        self.camera = camera
        self.time_next_frame = time_delay
        self.frame_rate = frame_rate


def gps2pixel_map(lat, lon):
    """
    42.526954, -90.726756 top left
    42.523530, -90.720147 bottom right
    image: 1233x870
    """
    top = 42.526954
    bottom = 42.523530
    left = -90.726756
    right = -90.720147
    width = 1233
    height = 870
    x_normalized = (lon - left) / (right - left)
    y_normalized = (lat - top) / (bottom - top)
    return x_normalized * width, y_normalized * height


def pixel2gps(H, x, y):
    gps_homogeneous = H.dot(np.array([x, y, 1]).T)
    gps_lat = gps_homogeneous[0] / gps_homogeneous[2]
    gps_lon = gps_homogeneous[1] / gps_homogeneous[2]
    return gps_lat, gps_lon


def get_calibration_path(sequence, camera):
    return f"AIC20_track3/{working_set}/S{sequence:02d}/c{camera:03d}/calibration.txt"


for camera in range(1, 6):
    print("SEQUENCE 1, CAMERA " + str(camera))
    directory_txt = get_calibration_path(sequence=1, camera=camera)
    h = read_homography(directory_txt)
    print(h)

# Trial with Sequence 1 Camera 1

sequence = 1
set_dict = {0: "train", 1: "test"}
working_set = set_dict[0]
algorithm_dict = {0: "mask_rcnn", 1: "ssd512", 2: "yolo3"}
algorithm = algorithm_dict[0]

camera = 1
end = 1955  # check folder cam_framenum

camera_dir = f"AIC20_track3/{working_set}/S{sequence:02d}/c{camera:03d}"
detec_dir = camera_dir + "/det/det_" + algorithm + ".txt"
detec_dir = "detections_mask_rcnn_ft_c001.txt"
gt = camera_dir + "/gt/gt.txt"
begin = 1
visualize = True

directory_txt = get_calibration_path(sequence=sequence, camera=camera)
H = read_homography(directory_txt)["homography"]
H = np.linalg.inv(H)  # Homography to go from pixel to gps coordinates

detections_list = Video(Video().getgroundtruth(detec_dir, end))  # 450

kalman_tracker = Sort()

# cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent',
#                         'Dark2', 'Set1', 'Set2', 'Set3',
#                         'tab10', 'tab20', 'tab20b', 'tab20c']

cmap = plt.cm.get_cmap("Dark2")

ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
plt.ion()
video_detections = []

for frame in detections_list.list_frames:
    print(frame.frame_id)
    detections = []
    frame_num = frame.frame_id

    path_now = camera_dir + "/frames/image_" + str(frame_num).zfill(3) + ".png"
    path_map = camera_dir + "/../reference_image.png"

    im = np.array(Image.open(path_now), dtype=np.uint8)
    im_map = np.array(Image.open(path_map), dtype=np.uint8)

    for box_detec in frame.bboxes:
        bbox = np.array(
            [
                box_detec.top_left[0],
                box_detec.top_left[1],
                box_detec.width + box_detec.top_left[0],
                box_detec.height + box_detec.top_left[1],
            ]
        )
        detections.append(bbox)

    trackers = kalman_tracker.update(np.array(detections))

    frame_detections = []

    for track_state in trackers:
        x, y = track_state[0], track_state[1]
        vx, vy = track_state[-4], track_state[-3]
        id = int(track_state[-1])
        bbox = convert_x_to_bbox(track_state[:-1])
        bbox = bbox.reshape(bbox.shape[1])

        if bbox[0] < 0:
            bbox[0] = 0

        track_det = np.concatenate((bbox, [id])).astype(np.uint64)

        frame_detections.append(
            [
                track_det[4],
                track_det[0],
                track_det[1],
                (track_det[2] - track_det[0]),
                (track_det[3] - track_det[1]),
            ]
        )

        num_color = cmap.N
        color = cmap(id % num_color)

        length = 10
        if np.abs(vx) + np.abs(vy) > 0.5:
            ax1.arrow(
                x,
                y,
                vx * length,
                vy * length,
                head_width=10,
                head_length=10,
                fc=color,
                ec=color,
            )
            bbox_width = track_det[2] - track_det[0]
            bbox_height = track_det[3] - track_det[1]
            detec = patches.Rectangle(
                (track_det[0], track_det[1]),
                bbox_width,
                bbox_height,
                linewidth=1.5,
                edgecolor=color,
                facecolor="none",
            )
            ax1.add_patch(detec)

            caption_id = "ID: {}".format(track_det[4])
            ax1.text(track_det[0], track_det[1] - 10, caption_id, color=color)

            # Calculate gps coordinates based on ground contact point.
            gps_lat, gps_lon = pixel2gps(H=H, x=x, y=y + bbox_height / 2)
            x_map, y_map = gps2pixel_map(lat=gps_lat, lon=gps_lon)
            print("x_map, y_map", x_map, y_map)
            square_size = 10
            detec2 = patches.Rectangle(
                (x_map - square_size / 2, y_map - square_size / 2),
                square_size,
                square_size,
                linewidth=1,
                edgecolor=color,
                fill=True,
                facecolor=color,
            )
            ax2.add_patch(detec2)
            ax2.text(x_map, y_map - 16, caption_id, color=color)

    ax1.text(10, 60, "Frame: {}".format(frame_num), color="black", fontsize=11)
    ax2.text(10, 60, "Frame: {}".format(frame_num), color="yellow", fontsize=11)

    video_detections.append(frame_detections)
    if visualize:
        ax1.axis("off")
        ax1.imshow(im)
        ax2.axis("off")
        ax2.imshow(im_map)
        plt.show()
        # plt.savefig("mtmc_images/image_"+str(frame_num).zfill(3))
        plt.pause(0.001)
        ax1.clear()
        ax2.clear()

gt_list = Video(Video().getgroundTruthown(gt, begin, end))

compute_IDF1(video_detections, gt_list)

print(compute_IDF1(video_detections, gt_list))
