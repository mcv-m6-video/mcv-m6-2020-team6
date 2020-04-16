import os
import cv2


def new_directory(path):
    if not os.path.exists(path):
            os.makedirs(path)


#new_directory('dataset/siamese_dataset')
directory_txt = 'annotation_only_cars.txt'
txt_gt = open(directory_txt, "r")

for line in txt_gt:
    splitLine = line.split(",")
    frameid = int(splitLine[0])
    x1 = float(splitLine[2])
    y1 = float(splitLine[3])
    width = float(splitLine[4])
    height = float(splitLine[5])
    id_track = int(splitLine[7])
    img = cv2.imread('dataset/frames/image'+str(frameid+1)+'.jpg')
    crop_img = img[round(y1):round(y1 + height), round(x1): round(x1 + width)]
    resize_img = cv2.resize(crop_img, (64, 64))
    new_directory('dataset/siamese_dataset/track_' + str(id_track))
    name = 'dataset/siamese_dataset/track_' + str(id_track)+'/image'+str(frameid+1)+'_track_'+str(id_track)+'.jpg'
    cv2.imwrite(name, resize_img)




