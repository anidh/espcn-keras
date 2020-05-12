# Keras-espcn
Keras implementation of Efficient Sub-Pixel Convolutional Neural Networks for super-resolution. The original paper is [Real-Time Single Image and Video Super-Resolution Using an Efficient
Sub-Pixel Convolutional Neural Network](https://arxiv.org/abs/1609.05158).

## Prerequisites
 * Tensorflow 1.12.0
 * keras 2.2.4
 * python 2.7.16
 * numpy
 * opencv
 * skimage

This code requires keras with a tensorflow backend. 

## Usage
run `python download_datasets.py` to download the datasets [BSD300](http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz) to the data folder, BSD300 contains training and test datasets
<br>
For training, run  
`python train.py --scale_factor=2`
<br>
For testing, run  
`python test.py --scale_factor=2`  
the Super-Resolution images are in the result folder

## Remark
When training the models with --scale_factor=1, it can be used for image enhancement such as deblur or mosaic Elimination and so on based on your train datasets, and you need to modify the data.py to meet your functional needs. For example, When scale_factor=1 this project change to a deblur net
<br>
Train for image enhancement, run  
`python train.py --scale_factor=1`
<br>
Test for image enhancement, run  
`python test.py --scale_factor=1`

## References
* [HighVoltageRocknRoll/sr](https://github.com/HighVoltageRocknRoll/sr) 


