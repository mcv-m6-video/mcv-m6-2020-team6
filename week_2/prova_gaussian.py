import os
import re
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

frames_dir = '../datasets/train/S03/c010/frames/'
roi_path = '../datasets/train/S03/c010/roi.jpg'

filelist = os.listdir(frames_dir)
filelist.sort(key=lambda f: int(re.sub('\D', '', f)))

img_list = []

for frame in filelist:
    img = glob.glob(frames_dir + frame)
    img_list.append(img[0])

# 25% training background
num_frames = len(img_list)

num_train = round(0.25 * num_frames)
num_test = round(num_frames - num_train)

train_list = img_list[:num_train]
test_list = img_list[num_train:]

ini_frame = cv2.imread(train_list[0])
image_list_bg = np.zeros((num_train, ini_frame.shape[0], ini_frame.shape[1]))

bar1 = tqdm(total=num_train)

for i in range(0, num_train):
    image = cv2.imread(train_list[i], cv2.IMREAD_GRAYSCALE)
    image_list_bg[i, :, :] = image
    bar1.update(1)

bar1.close()

mean = image_list_bg.mean(axis=0)
std = image_list_bg.std(axis=0)

visualize = False

if visualize:
    plt.imshow(mean, cmap="gray")
    plt.axis('off')
    plt.show()
    plt.imshow(std, cmap="gray")
    plt.axis('off')
    plt.show()


def frame_mask_single_gaussian(img, mean_model, std_model, alpha):
    foreground = abs(img - mean_model) >= alpha * (std_model + 2)
    return foreground


bar2 = tqdm(total=num_test)
image_list_fg = np.zeros((num_test, ini_frame.shape[0], ini_frame.shape[1]))
roi = cv2.cvtColor(cv2.imread(roi_path), cv2.COLOR_BGR2GRAY)
#plt.imshow(roi, cmap='gray')
#plt.show()


for j in range(0, num_test):
    image = cv2.imread(test_list[j], cv2.IMREAD_GRAYSCALE)
    image_list_fg[j, :, :] = roi*frame_mask_single_gaussian(image, mean, std, 1)
    bar2.update(1)
bar2.close()
plt.imshow(image_list_fg[0], cmap='gray')
plt.show()

"""
img = cv2.imread(train_list[0],cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap="gray")
plt.axis('off')
plt.show()

hist,bins = np.histogram(img.flatten(),256,[0,256])
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.show()

# create a CLAHE object (Arguments are optional).
clahe = cv2.createCLAHE()
cl1 = clahe.apply(img)

plt.imshow(cl1,cmap="gray")
plt.show()

hist,bins = np.histogram(cl1.flatten(),256,[0,256])
plt.hist(cl1.flatten(),256,[0,256], color = 'b')
plt.xlim([0,256])
plt.show()
#cv2.imwrite('prova.jpg',cl1)
"""
