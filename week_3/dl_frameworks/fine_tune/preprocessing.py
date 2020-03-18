import glob
import os
import re
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.patches as patches


def resize_images(frames_dir):

    """Resize images from 11920x1080 to 500x281"""

    filelist = os.listdir(frames_dir)
    filelist.sort(key=lambda f: int(re.sub('\D', '', f)))
    for frame in filelist:
        img = glob.glob(frames_dir + frame)
        num_image = frame.split('.')
        num_frame = num_image[0].split('e')[1]
        image = Image.open(img[0])
        image.thumbnail((500, 500))
        image.save('images/images/image' + num_frame + '.jpg')


def resize_boxes(annot_dir, original_image, resize_image):

    """Resize bounding boxes for the new resize images 500x281"""

    image_ori = Image.open(original_image)
    image_res = Image.open(resize_image)
    ratio = image_ori.size[0] / image_res.size[0]
    old_frame_id = 0
    directory_txt = annot_dir
    gt = []
    txt_gt = open(directory_txt, "r")
    for line in txt_gt:
        splitLine = line.split(",")
        frameid = int(splitLine[0])
        if frameid == old_frame_id:
            x1 = float(splitLine[2]) / ratio
            y1 = float(splitLine[3]) / ratio
            width = float(splitLine[4]) / ratio
            height = float(splitLine[5]) / ratio
            gt.append(str(x1) + ',' + str(y1) + ',' + str(width) + ',' + str(height))
        else:
            old_frame_id = frameid
            gt = []
        gt_frame = open('images/annots/image' + str(frameid + 1) + '.txt', 'w')
        for i in gt:
            gt_frame.writelines(i + '\n')
        gt_frame.close()




def check_resize_process(image, directory_txt):

    """Check resize image and corresponding bounding boxes"""

    fig, ax = plt.subplots(1)

    ax.imshow(image)

    txt_gt = open(directory_txt, "r")
    gt=[]
    for line in txt_gt:
        splitLine = line.split(",")
        x1 = float(splitLine[0])
        y1 = float(splitLine[1])
        width = float(splitLine[2])
        height = float(splitLine[3])
        gt.append([x1, y1, width, height])
    for box in gt:
        # Create a Rectangle patch
        rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    plt.show()


#frames_dir = 'C:/Users/Quim/Desktop/frames/'
#resize_images(frames_dir)

#original_image = 'C:/Users/Quim/Desktop/frames/image1.jpg'
#resize_image = 'images/images/image1.jpg'

#annot_dir = 'annotation_only_cars.txt'
#resize_boxes(annot_dir, original_image, resize_image)


#image = Image.open('images/images/image1000.jpg')
#gt = 'images/annots/image1000.txt'
#check_resize_process(image, gt)