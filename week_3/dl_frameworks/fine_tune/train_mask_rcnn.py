from Ai_city_mask_rcnn import *


def training_fine_tune(dataset_dir):

    train_set = Aicity_Dataset()
    train_set.load_dataset(dataset_dir, is_train=True)
    train_set.prepare()
    print('Train: %d' % len(train_set.image_ids))

    # test/val set
    test_set = Aicity_Dataset()
    test_set.load_dataset(dataset_dir, is_train=False)
    test_set.prepare()
    print('Test: %d' % len(test_set.image_ids))

    config = AicityConfig()
    config.display()

    # define the model
    model = MaskRCNN(mode='training', config=config, model_dir='logs/')

    # load weights (mscoco) and exclude the output layers
    model.load_weights('weights/mask_rcnn_coco.h5', by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    # train weights (output layers or 'heads')
    model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=250, layers='heads')


#dataset_dir = 'images/'
#training_fine_tune(dataset_dir)


