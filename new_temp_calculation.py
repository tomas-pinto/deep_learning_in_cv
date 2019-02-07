## LOAD CAMVID ##
import keras
import random
import numpy as np

from Utils.sequence import generate_data
from Utils.load_camvid import make_x2y, make_rgb2label, make_set
from Utils.model import calibration_softmax, relu6, mobilenetV2
from Utils.new_temp_scaling import TemperatureScaling

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
model = mobilenetV2(input_shape=(720, 960, 3), classes=12, alpha=1., reg=0.0, d=0.0)
model.summary()

## LOAD WEIGHTS ##
name = 'baseline_weights' #Change this name to load the best model
model.load_weights("./Weights/{}.h5".format(name))

## POP LAST LAYER ##
model.layers.pop()
i = model.input
o = model.layers[-1].output
model = keras.models.Model(inputs=i, outputs=[o])

generator = generate_data(val_files[0:2],2,x2y,rgb2label,x_dir,y_dir)

# Find temperature by minimizing NLL Loss
a = TemperatureScaling(model)
a.fit(generator)
