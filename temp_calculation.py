## LOAD CAMVID ##
import keras
import random
import numpy as np

from sequence import generate_data
from load_camvid import make_x2y, make_rgb2label, make_set
from model import calibration_softmax, relu6, mobilenetV2
from temp_scaling import TemperatureScaling

# Create a Dictionary of filenames that maps input x to labels y
x_dir = './ML-Datasets/CamVid/701_StillsRaw_full'
y_dir = './ML-Datasets/CamVid/LabeledApproved_full'
x2y = make_x2y(x_dir,y_dir)

# Create a Dictionary to convert rgb data to labeled data
rgb2label = make_rgb2label('./ML-Datasets/CamVid/label_colors.txt')

# Get the training, validation and test sets
train_files = make_set("./ML-Datasets/CamVid/train.txt")
val_files = make_set("./ML-Datasets/CamVid/val.txt")
test_files = make_set("./ML-Datasets/CamVid/test.txt")

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

# Define the batch to analyze
files = test_files + val_files

# Shuffle Test+Val sets
random.shuffle(files)

# Define the batch to analyze
file = files[0:31]
batch_size = len(file)

_,y = generate_data(file,batch_size,x2y,rgb2label,x_dir,y_dir).__getitem__(0)
print(y.shape)

prediction = model.predict_generator(generate_data(file,1,x2y,rgb2label,x_dir,y_dir))
print(prediction.shape)

# flatten logits and ground truth
prediction = np.reshape(prediction,(batch_size*720*960,12))
y = np.reshape(y,(batch_size*720*960,12))

print(y.shape)
print(prediction.shape)

# Find temperature by minimizing NLL Loss
a = TemperatureScaling(model)
a.fit(prediction,y)