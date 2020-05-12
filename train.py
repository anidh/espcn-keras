# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
import data as dg
import model as espcn_model
import tensorflow as tf
import tensorflow.compat.v1.keras.backend as K
import tensorflow_model_optimization as tfmot


parser = argparse.ArgumentParser()
parser.add_argument('--model', default='espcn', type=str, help='select the model save address')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--train_data', default='data/DIV2K/train', type=str, help='path of train data')
parser.add_argument('--epoch', default=1000, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
parser.add_argument('--save_every', default=5, type=int, help='save model at every x epoches')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument('--scale_factor', default=4, type=int, help='scale_factor')
args = parser.parse_args()

save_dir = os.path.join('models',args.model,'saved_weights')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# set the GPU parameters
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True 
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)

# dynamically adjust the learning rate
def lr_schedule(epoch):
    initial_lr = args.lr
    if epoch <= 100:
        lr = initial_lr
    else:
        lr = initial_lr / 10
    return lr

# the generator of train datasets for fit_generator of keras
def train_datagen(epoch_num=5, batch_size=32, data_dir=args.train_data):
    n_count = 0
    while(True):
        if n_count == 0:
            xs,ys = dg.datagenerator(data_dir, args.scale_factor)

            #normalized
            xs = xs.astype('float32') / 255.0
            ys = ys.astype('float32') / 255.0
            indices = list(range(xs.shape[0]))
            n_count = 1
        for _ in range(epoch_num):
            #shuffle
            np.random.shuffle(indices)    
            for i in range(0, len(indices), batch_size):
                batch_x = xs[indices[i:i + batch_size]]
                batch_y = ys[indices[i:i + batch_size]]
                yield batch_x, batch_y


if __name__ == '__main__':
    # get the espcn model
    espcn = espcn_model.espcn(scale_factor=args.scale_factor)
    model = espcn()
    
    #Printing the summary of the model to see that it doesn't contain quant aware layers\
    model.summary()

    model.compile(optimizer=Adam(args.lr), loss='mse')

    #set keras callback function to save the model
    checkpointer = ModelCheckpoint(os.path.join(save_dir,'model_weights_{epoch:03d}.h5'),
                 verbose=1, save_weights_only=True,save_freq='epoch', period=args.save_every)

    # set keras callback function to dynamically adjust the learning rate
    lr_scheduler = LearningRateScheduler(lr_schedule)

    # start train
    model.fit(train_datagen(batch_size=args.batch_size), epochs=args.epoch, verbose=1, steps_per_epoch = 300,callbacks=[lr_scheduler,checkpointer])




















