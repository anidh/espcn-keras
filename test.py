# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import cv2
from keras.models import load_model
from skimage.io import imsave
import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--set_dir', default='data/BSDS300/images/test', type=str, help='directory of test dataset')
    parser.add_argument('--model_dir', default=os.path.join('models','espcn'), type=str, help='directory of the model')
    parser.add_argument('--model_name', default='model_300.hdf5', type=str, help='the model name')
    parser.add_argument('--result_dir', default='results', type=str, help='directory of results')
    parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
    parser.add_argument('--scale_factor', default=2, type=int, help='scale_factor')
    return parser.parse_args()

args = parse_args()
# definition the sub_pixel function because we use Lambda in the model
scale_factor = args.scale_factor
def sub_pixel(x):
    return tf.depth_to_space(x, scale_factor)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # load the model
    if scale_factor > 1:
        model = load_model(os.path.join(args.model_dir, args.model_name),custom_objects={"sub_pixel": sub_pixel}, compile=False)
    else:
        model = load_model(os.path.join(args.model_dir, args.model_name), compile=False)
    i = 0
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)
    for file_name in os.listdir(args.set_dir):
	print(file_name)

        # read the input image
        img = cv2.imread(os.path.join(args.set_dir,file_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img, dtype='uint8')

        # expand the batchsize dimension
        img = np.expand_dims(img,axis=0)

        # normalized
        img = img.astype('float32') / 255.0

        # run the model
        output = model.predict(img)

        # squeeze the batchsize dimension
        output = np.squeeze(output, 0)

        # save the super resolution image
        i = i + 1
        pic = str(i)+".png"
        result = os.path.join(args.result_dir, pic)
        imsave(result, np.clip(output,0,1))

        
        


