from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import keras
import pandas as pd
import tensorflow as tf

from keras import backend as K
from keras.models import load_model
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler

from Utils.load_camvid import make_x2y, make_rgb2label, make_set
from Utils.sequence import generate_data
from Utils.model import relu6, mobilenetV2
from Utils.losses import weighted_categorical_crossentropy, weighted_dirichlet_loss
from Utils.metrics import MeanIoU, IoU, single_class_accuracy

## LOAD CAMVID ##
# Create a Dictionary of filenames that maps input x to labels y
x_dir = './CamVid/701_StillsRaw_full'
y_dir = './CamVid/LabeledApproved_full'
x2y = make_x2y(x_dir,y_dir)

# Create a Dictionary to convert rgb data to labeled data
rgb2label = make_rgb2label('./CamVid/label_colors.txt')

# Get the training, validation and test sets
train_files = make_set("./CamVid/train.txt")
val_files = make_set("./CamVid/val.txt")
test_files = make_set("./CamVid/test.txt")

## BUILD MODEL ##
if sys.argv[1] == 'data_augmentation':
    model = mobilenetV2(input_shape=(240, 240, 3), classes=12, alpha=1., reg=sys.argv[2], d=sys.argv[3])
    generator = generate_data(train_files,3,x2y,x_dir, y_dir, dirichlet=False, data_aug=True)
    initial_lr = 0.001
    update = 0.995

else if sys.argv[2] == 'finetuning':
    model = mobilenetV2(input_shape=(720, 960, 3), classes=12, alpha=1., reg=sys.argv[2], d=sys.argv[3])
    generator = generate_data(train_files,3,x2y,x_dir, y_dir, dirichlet=False, data_aug=False)
    initial_lr = 0.0001
    update = 1

model.summary()

## DEFINE METRICS AND WEIGHTS LOSS FUNCTION ##
num_classes = 12
miou_metric = MeanIoU(num_classes)
void_iou_metric = IoU(num_classes,0)
sky_iou_metric = IoU(num_classes,1)
building_iou_metric = IoU(num_classes,2)
pole_iou_metric = IoU(num_classes,3)
road_iou_metric = IoU(num_classes,4)
pavement_iou_metric = IoU(num_classes,5)
tree_iou_metric = IoU(num_classes,6)
sign_iou_metric = IoU(num_classes,7)
fence_iou_metric = IoU(num_classes,8)
car_iou_metric = IoU(num_classes,9)
pedestrian_iou_metric = IoU(num_classes,10)
cyclist_iou_metric = IoU(num_classes,11)

void_acc_metric = single_class_accuracy(0)
sky_acc_metric = single_class_accuracy(1)
building_acc_metric = single_class_accuracy(2)
pole_acc_metric = single_class_accuracy(3)
road_acc_metric = single_class_accuracy(4)
pavement_acc_metric = single_class_accuracy(5)
tree_acc_metric = single_class_accuracy(6)
sign_acc_metric = single_class_accuracy(7)
fence_acc_metric = single_class_accuracy(8)
car_acc_metric = single_class_accuracy(9)
pedestrian_acc_metric = single_class_accuracy(10)
cyclist_acc_metric = single_class_accuracy(11)

# weights when using median frequency balancing used in SegNet paper
# https://arxiv.org/pdf/1511.00561.pdf
# The numbers were generated by:
# https://github.com/yandex/segnet-torch/blob/master/datasets/camvid-gen.lua

weights = np.array([0.0,
                    0.58872014284134,
                    0.51052379608154,
                    2.6966278553009,
                    0.45021694898605,
                    1.1785038709641,
                    0.77028578519821,
                    2.4782588481903,
                    2.5273461341858,
                    1.0122526884079,
                    3.2375309467316,
                    4.1312313079834])

if sys.argv[4] == 'dirichlet':
    chosen_loss = weighted_dirichlet_loss(weights)
    generator.dirichlet = True

else if sys.argv[4] == 'crossentropy':
    chosen_loss = weighted_categorical_crossentropy(weights)

## COMPILE MODEL ##
# Define Loss and Optimizer
model.compile(
    loss=chosen_loss,
    optimizer=keras.optimizers.Adam(lr=initial_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
    metrics=['accuracy',
             miou_metric.mean_iou,
             void_iou_metric.iou,
             sky_iou_metric.iou,
             building_iou_metric.iou,
             pole_iou_metric.iou,
             road_iou_metric.iou,
             pavement_iou_metric.iou,
             tree_iou_metric.iou,
             sign_iou_metric.iou,
             fence_iou_metric.iou,
             car_iou_metric.iou,
             pedestrian_iou_metric.iou,
             cyclist_iou_metric.iou,
             void_acc_metric,
             sky_acc_metric,
             building_acc_metric,
             pole_acc_metric,
             road_acc_metric,
             pavement_acc_metric,
             tree_acc_metric,
             sign_acc_metric,
             fence_acc_metric,
             car_acc_metric,
             pedestrian_acc_metric,
             cyclist_acc_metric])

## LOAD MODEL ##
if len(sys.argv) == 5:
    weight_name = sys.argv[5]
    model.load_weights("./Weights/{}.h5".format(weight_name))

## START TRAINING ##
# Define Callbacks
filepath = "./Logs/deeplabv3-{epoch:02d}-{val_mean_iou:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=True, mode='auto', period=1)
csv_logger = CSVLogger("./Logs/logs.csv", append=True, separator=';')

# Define Training
model.fit_generator(
    generate_data(train_files,3,x2y,rgb2label),
    epochs = 150,
    verbose = 1,
    callbacks = [checkpoint, csv_logger, LearningRateScheduler(lambda epoch:  initial_lr * update ** (epoch), verbose = 1)],
    validation_data=generate_data(val_files,3,x2y,rgb2label))
