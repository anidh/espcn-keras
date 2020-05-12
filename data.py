# -*- coding: utf-8 -*-
import glob
import cv2
import numpy as np

# training data preprocessing, the output image pair size is 128x128 and 256x256
def gen_patches(file_name, scale_factor=4):
    # read image
    img = cv2.imread(file_name)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:, :, 0]
    if len(img.shape) < 3:
        img = img[:, :, np.newaxis]
    h, w = img.shape[0:2]
    scale = 256 // scale_factor

    # crop image size as 256x256
    a = int(h // 2 - 128)
    b = int(h // 2 + 128)
    c = int(w // 2 - 128)
    d = int(w // 2 + 128)
    img = img[a:b, c:d]

    if scale_factor == 1:
        # Gaussian blur
        img_down = cv2.GaussianBlur(img, (3, 3), 0.5)
    else:
        # resize the image
        img_down = cv2.resize(img, (scale, scale))[:, :, np.newaxis]

    return img_down, img

# load all training data into array
def datagenerator(data_dir ='./train/', scale_factor=4):
    file_list = glob.glob(data_dir+'/*.png')
    data = []
    data_down = []
    for i in range(len(file_list)):
        img_down, img = gen_patches(file_list[i], scale_factor)
        data.append(img)
        data_down.append(img_down)
    data = np.array(data, dtype='uint8')
    data_down = np.array(data_down, dtype='uint8')
    return data_down, data
