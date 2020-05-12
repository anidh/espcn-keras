# -*- coding: utf-8 -*-
import numpy as np
from tensorflow.keras.layers import  Input, Conv2D, Lambda
from tensorflow.keras.models import Model,load_model
import tensorflow as tf
import tensorflow_model_optimization as tfmot


class espcn:
    def __init__(self, scale_factor=4, image_channels=1,loader=False, qat=True):
        self.__name__ = 'espcn'
        self.scale_factor = scale_factor
        self.channels = image_channels
        self.loader = loader

    # upsampling the resolution of image
    def sub_pixel(self, x):
        return tf.compat.v1.depth_to_space(x, self.scale_factor,name="prediction")
        
    # building the espcn network
    def __call__(self):
        if self.loader is True:
            input_image = Input(shape=(240, 432, self.channels), name='x')
        else:    
            input_image = Input(shape=(None, None, self.channels), name='x')
        x = Conv2D(32, 5, kernel_initializer='glorot_uniform', padding='same', activation=tf.nn.relu,name="conv1")(input_image)
        x = Conv2D(32, 3, kernel_initializer='glorot_uniform', padding='same',activation=tf.nn.relu,name="conv2")(x)
        x = Conv2D(self.scale_factor**2*self.channels, 3, kernel_initializer='glorot_uniform', padding='same',activation=tf.nn.relu,name="conv3")(x)
        if self.scale_factor > 1:
            y = self.sub_pixel(x)
        model = Model(inputs=input_image, outputs=y)
        return model
