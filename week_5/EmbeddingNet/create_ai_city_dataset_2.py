import os
import cv2


def new_directory(path):
    if not os.path.exists(path):
            os.makedirs(path)


camera='c003'
scene='S01'
i =0

directory_txt = 'D:/Documents/ai_city_challenge/dataset/'+scene+'/'+camera+'/gt/gt.txt'
with open(directory_txt, 'r') as f:
    lines = f.readlines()

new_list = []
for line in lines:
    splitLine = line.split(",")
    frameid = int(splitLine[0])
    x1 = int(splitLine[2])
    y1 = int(splitLine[3])
    width = int(splitLine[4])
    height = int(splitLine[5])
    id_track = int(splitLine[1])
    new_list.append(str(frameid) + ',' +str(id_track) + ',' +str(x1) + ',' +str(y1) + ',' +str(width) + ',' +str(height))

tracks = sorted(new_list, key=lambda x: int(x.split(',')[1]))
directory_txt_1 = 'D:/Documents/ai_city_challenge/dataset/'+scene+'/'+camera+'/gt/gt_1.txt'
detec_file = open(directory_txt_1, 'w')

for x in tracks:
    detec_file.writelines(x + '\n')
detec_file.close()


#new_directory('Dataset/prova')
txt_gt = open(directory_txt_1, "r")

firstLine = txt_gt.readline()
id_track_old = int(firstLine.split(",")[1])

j = 0

for line in txt_gt:
    splitLine = line.split(",")
    frameid = int(splitLine[0])
    x1 = int(splitLine[2])
    y1 = int(splitLine[3])
    width = int(splitLine[4])
    height = int(splitLine[5])
    id_track = int(splitLine[1])

    if id_track != id_track_old:
        i += 1
        j = 1
    if id_track == id_track_old:
        j += 1

    img = cv2.imread('D:/Documents/ai_city_challenge/dataset/'+scene+'/'+camera+'/frames/image'+str(frameid).zfill(5)+'.jpg')
    crop_img = img[round(y1):round(y1 + height), round(x1): round(x1 + width)]
    resize_img = cv2.resize(crop_img, (64, 64))
    new_directory('Dataset/train/class_' + str(i))
    name = 'Dataset/train/class_' + str(i) + '/image' + str(j) + '_track_' + str(i) + '.jpg'
    cv2.imwrite(name, resize_img)
    id_track_old = id_track
    print(name)




