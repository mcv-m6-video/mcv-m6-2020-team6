from preprocessing import *
from Ai_city_visualization import *
from train_mask_rcnn import *
from test_mask_rcnn import *


def preprocessing():
    frames_dir = 'C:/Users/Quim/Desktop/frames/'
    resize_images(frames_dir)

    original_image = 'C:/Users/Quim/Desktop/frames/image1.jpg'
    resize_image = 'images/images/image1.jpg'

    annot_dir = 'annotation_only_cars.txt'
    resize_boxes(annot_dir, original_image, resize_image)

    image = Image.open('images/images/image1000.jpg')
    gt = 'images/annots/image1000.txt'
    check_resize_process(image, gt)


def data_visualization():
    dataset_dir = 'images/'
    mask_visualization(dataset_dir)
    instance_seg_gt_visualization(dataset_dir, 1)


def training_fine_tune():
    dataset_dir = 'images/'
    training_fine_tune(dataset_dir)


def test_fine_tune():
    dataset_dir = 'images/'
    test_eval_fine_tune(dataset_dir)
    extract_detections_fine_tune(dataset_dir)


if __name__ == '__main__':
    preprocessing()
    # data_visualization()
    # training_fine_tune()
    # test_fine_tune()
