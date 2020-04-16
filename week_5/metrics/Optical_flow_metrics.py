import cv2
import numpy as np
from matplotlib import pyplot as plt


# based on flow development kit of kitti web
def flow_read(gt_dir, test_dir):
    # cv2 imread ---> BGR  need to converted in RGB format

    gt_kitti = cv2.imread(gt_dir, -1)

    test_kitti = cv2.imread(test_dir, -1)

    u_test = (test_kitti[:, :, 2] - 2.0 ** 15) / 64

    v_test = (test_kitti[:, :, 1] - 2.0 ** 15) / 64

    valid_test = test_kitti[:, :, 0]

    u_gt = (gt_kitti[:, :, 2] - 2.0 ** 15) / 64

    v_gt = (gt_kitti[:, :, 1] - 2.0 ** 15) / 64

    valid_gt = gt_kitti[:, :, 0]

    F_gt = np.transpose(np.array([u_gt, v_gt, valid_gt]))

    F_test = np.transpose(np.array([u_test, v_test, valid_test]))

    return F_gt, F_test


def Flow_read(flow_dir):
    # cv2 imread ---> BGR  need to converted in RGB format

    im = cv2.imread(flow_dir, cv2.IMREAD_UNCHANGED)
    im_kitti = np.flip(im, axis=2).astype(np.double)

    u_f = (im_kitti[:, :, 0] - 2.0 ** 15) / 64
    v_f = (im_kitti[:, :, 1] - 2.0 ** 15) / 64

    f_valid = im_kitti[:, :, 2]
    f_valid[f_valid > 1] = 1

    u_f[f_valid == 0] = 0
    v_f[f_valid == 0] = 0

    flow = np.dstack([u_f, v_f])

    flow_1 = np.dstack([u_f, v_f, f_valid])

    return flow, flow_1


def msen(F_gt, F_test):
    SEN = []

    E_du = F_gt[:, :, 0] - F_test[:, :, 0]

    E_dv = F_gt[:, :, 1] - F_test[:, :, 1]

    E = np.sqrt(E_du ** 2 + E_dv ** 2)

    F_valid_gt = F_gt[:, :, 2]

    E[F_valid_gt == 0] = 0  # 0s in ocluded pixels

    SEN = np.append(
        SEN, E[F_valid_gt != 0]
    )  # take in account the error of the non-ocluded pixels

    MSEN = np.mean(SEN)

    plt.figure(1)
    plt.hist(E[F_valid_gt == 1], bins=40, density=True)
    plt.title("Optical Flow error")
    plt.xlabel("MSEN")
    plt.ylabel("Number of pixels")
    plt.savefig("histogram.png")
    plt.show()

    plt.figure(2)
    plt.imshow(np.reshape(E, F_gt.shape[:-1]))
    plt.colorbar()
    plt.tick_params(axis="both", labelbottom=False, labelleft=False)
    plt.savefig("error_image.png")
    plt.show()

    return MSEN


def pepn(F_gt, F_test, th):
    SEN = []

    E_du = F_gt[:, :, 0] - F_test[:, :, 0]

    E_dv = F_gt[:, :, 1] - F_test[:, :, 1]

    E = np.sqrt(E_du ** 2 + E_dv ** 2)

    F_valid_gt = F_gt[:, :, 2]

    E[F_valid_gt == 0] = 0  # 0s in ocluded pixels

    SEN = np.append(
        SEN, E[F_valid_gt != 0]
    )  # take in account the error of the non-ocluded pixels

    PEPN = (np.sum(SEN > th) / len(SEN)) * 100

    return PEPN


"""
gt_dir1 = "D:\\Documents\\Proves\\week1\\datasets\\kitti\\groundtruth\\000045_10.png"
test_dir1 = 'D:\\Documents\\Proves\\week1\\datasets\\kitti\\results\\LKflow_000045_10.png'

F_gt1, F_test1 = flow_read(gt_dir1,test_dir1)

MSEN1 = msen(F_gt1, F_test1)
PEPN1 = pepn(F_gt1, F_test1, 3)

print(MSEN1)
print(PEPN1)

gt_dir2 = "D:\\Documents\\Proves\\week1\\datasets\\kitti\\groundtruth\\000157_10.png"
test_dir2 = 'D:\\Documents\\Proves\\week1\\datasets\\kitti\\results\\LKflow_000157_10.png'

F_gt2, F_test2 = flow_read(gt_dir2,test_dir2)

MSEN2 = msen(F_gt2, F_test2)
PEPN2 = pepn(F_gt2, F_test2, 3)

print(MSEN2)
print(PEPN2)

"""
