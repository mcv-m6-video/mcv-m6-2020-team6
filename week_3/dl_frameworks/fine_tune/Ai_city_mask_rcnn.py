from mrcnn.utils import Dataset
import re
import glob
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
import skimage.io


class Aicity_Dataset(Dataset):

    # load the dataset definitions
    def load_dataset(self, dataset_dir, is_train=True):

        # define one class
        self.add_class("dataset", 1, "Car")

        # define data locations
        images_dir = dataset_dir + 'images/'
        annotations_dir = dataset_dir + 'annots/'

        filelist = os.listdir(images_dir)
        filelist.sort(key=lambda f: int(re.sub('\D', '', f)))
        img_list = []

        for frame in filelist:
            img = glob.glob(images_dir + frame)
            img_list.append(img[0])

        num_images = len(img_list)

        # find all images
        for filename in img_list:
            name = filename.split('/')
            image = name[2].split('.')
            image_id = image[0].split('e')[1]

            # skip all images after 535 if we are building the train set
            if is_train and int(image_id) >= round(0.25 * num_images) + 1:
                continue

            # skip all images before 535 if we are building the test/val set
            if not is_train and int(image_id) < round(0.25 * num_images) + 1:
                continue

            img_path = images_dir + image[0] + '.jpg'
            ann_path = annotations_dir + 'image' + image_id + '.txt'

            # add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    # extract bounding boxes from an txt annotation file
    @staticmethod
    def extract_boxes(filename):
        height = []
        width = []
        top_left = []

        txt_gt = open(filename, "r")
        for line in txt_gt:
            splitLine = line.split(",")
            x1 = float(splitLine[0])
            y1 = float(splitLine[1])
            top_left_1 = [x1, y1]
            width_1 = float(splitLine[2])
            height_1 = float(splitLine[3])

            top_left.append(top_left_1)
            width.append(width_1)
            height.append(height_1)

        txt_gt.close()

        return top_left, width, height

    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        # load XML
        top_left, w, h = self.extract_boxes(path)
        # create one array for all masks, each on a different channel

        resize_image = 'images/images/image1.jpg'
        image_res = Image.open(resize_image)

        masks = np.zeros([image_res.size[1], image_res.size[0], len(top_left)], dtype='uint8')
        # create masks
        class_ids = list()
        for i in range(len(top_left)):
            box = top_left[i]
            width = w[i]
            height = h[i]
            masks[round(box[1]):round(box[1] + height), round(box[0]):round(box[0] + width), i] = 1
            class_ids.append(self.class_names.index('Car'))
        return masks, np.asarray(class_ids, dtype='int32')

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


class AicityConfig(Config):
    NAME = "Aicity_cfg"
    NUM_CLASSES = 1 + 1
    STEPS_PER_EPOCH = 150


class PredictionConfig(Config):
    NAME = "Ai_city_cfg"
    NUM_CLASSES = 1 + 1
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
