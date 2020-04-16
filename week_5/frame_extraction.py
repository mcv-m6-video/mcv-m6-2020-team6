import os

import cv2


def frame_extraction_cv2(source, frame_folder):
    video_source = source + "vdo.avi"
    video = cv2.VideoCapture(video_source)
    success, image = video.read()
    count = 1
    if not os.path.exists(frame_folder):
        os.mkdir(frame_folder)
    while success:

        cv2.imwrite(
            frame_folder + "/image%05d.jpg" % count, image
        )  # save frame as JPEG file
        success, image = video.read()
        print("Read a new frame: ", success)
        count += 1


frame_folder = "dataset/c005/frames"
folder = "D:/Documents/ai_city_challenge/dataset/c005/"

frame_extraction_cv2(folder, frame_folder)
