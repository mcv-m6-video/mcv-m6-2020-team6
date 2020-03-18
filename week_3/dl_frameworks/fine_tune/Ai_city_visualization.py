from Ai_city_mask_rcnn import *


def mask_visualization(dataset_dir):
    train_set = Aicity_Dataset()
    train_set.load_dataset(dataset_dir, is_train=True)
    train_set.prepare()
    print('Train: %d' % len(train_set.image_ids))

    test_set = Aicity_Dataset()
    test_set.load_dataset(dataset_dir, is_train=False)
    test_set.prepare()
    print('Test: %d' % len(test_set.image_ids))

    for image_id in train_set.image_ids:
        # load image info
        info = train_set.image_info[image_id]
        # display on the console
        print(info)

    image = train_set.load_image(1)
    plt.imshow(image)
    # plot all masks
    mask, _ = train_set.load_mask(1)
    for j in range(mask.shape[2]):
        plt.axis('off')
        plt.imshow(mask[:, :, j], cmap='gray', alpha=0.25)
    # show the figure
    plt.show()


def instance_seg_gt_visualization(dataset_dir, frame_id):
    train_set = Aicity_Dataset()
    train_set.load_dataset(dataset_dir, is_train=True)
    train_set.prepare()
    print('Train: %d' % len(train_set.image_ids))

    test_set = Aicity_Dataset()
    test_set.load_dataset(dataset_dir, is_train=False)
    test_set.prepare()
    print('Test: %d' % len(test_set.image_ids))

    image_id = frame_id
    image = train_set.load_image(image_id)
    mask, class_ids = train_set.load_mask(image_id)
    bbox = extract_bboxes(mask)
    display_instances(image, bbox, mask, class_ids, train_set.class_names)


#dataset_dir = 'images/'
#mask_visualization(dataset_dir)
#instance_seg_gt_visualization(dataset_dir, 1)