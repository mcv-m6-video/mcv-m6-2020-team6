import skimage.io
from Ai_city_mask_rcnn import *


def evaluate_model(dataset, model, cfg):
    APs = list()
    for image_id in dataset.image_ids:
        # load image, bounding boxes and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = np.expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)
        # extract results for first sample
        r = yhat[0]
        # calculate statistics, including AP
        AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
        # store
        APs.append(AP)
    # calculate the mean AP across all images
    mAP = np.mean(APs)
    return mAP


def test_eval_fine_tune(dataset_dir):

    train_set = Aicity_Dataset()
    train_set.load_dataset(dataset_dir, is_train=True)
    train_set.prepare()
    print('Train: %d' % len(train_set.image_ids))

    test_set = Aicity_Dataset()
    test_set.load_dataset(dataset_dir, is_train=False)
    test_set.prepare()
    print('Test: %d' % len(test_set.image_ids))

    cfg = PredictionConfig()
    model = MaskRCNN(mode='inference', model_dir='predictions/', config=cfg)
    model.load_weights('mask_rcnn_aicity_cfg_0037.h5', by_name=True)
    train_mAP = evaluate_model(train_set, model, cfg)
    print("Train mAP: %.3f" % train_mAP)
    test_mAP = evaluate_model(test_set, model, cfg)
    print("Test mAP: %.3f" % test_mAP)


def extract_detections_fine_tune(dataset_dir):
    cfg = PredictionConfig()
    model = MaskRCNN(mode='inference', model_dir='predictions/', config=cfg)
    model.load_weights('mask_rcnn_aicity_cfg_0037.h5', by_name=True)

    filelist = os.listdir(dataset_dir + 'images/')
    filelist.sort(key=lambda f: int(re.sub('\D', '', f)))
    detect_list = []
    ratio = 3.84
    for frame in filelist:
        img = glob.glob(dataset_dir + 'images/' + frame)

        name = img[0].split('/')

        image = name[-1].split('.')
        frame_id = image[0].split('e')[1]

        image = skimage.io.imread(img[0])

        # Run detection
        results = model.detect([image], verbose=1)

        # Visualize results
        r = results[0]
        for box, score, label in zip(r['rois'], r['scores'], r['class_ids']):
            # if label == 2 or label == 3 or label == 4 or label == 6 or label == 8:
            detect_list.append(
                str(frame_id) + ',' + str(label) + ',' + str(box[1] * ratio) + ',' + str(box[0] * ratio) + ','
                + str((box[3] * ratio - box[1] * ratio + 1)) + ',' + str(
                    (box[2] * ratio - box[0] * ratio + 1)) + ',' + str(
                    score) + ',' + str(-1) + ',' + str(-1)
                + ',' + str(-1))

    detections = sorted(detect_list, key=lambda x: int(x.split(',')[0]))
    detec_file = open('detections_mask_rcnn_fine_tune.txt', 'w')

    for i in detections:
        detec_file.writelines(i + '\n')
    detec_file.close()

#dataset_dir = 'images/'
#test_eval_fine_tune(dataset_dir)
#extract_detections_fine_tune(dataset_dir)