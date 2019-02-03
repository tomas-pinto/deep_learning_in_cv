import sys
import keras
import numpy as np

from load_camvid import make_x2y, make_rgb2label, make_set
from sequence import generate_data
from model import calibration_softmax, relu6, mobilenetV2
from losses import weighted_categorical_crossentropy

## HOW TO CALL ##
# Type in cmd (first for temperature, second for dirichlet):
# python calibration_analysis.py 't' float(desired_T) 'baseline_weights'
# python calibration_analysis.py 'd' 'dirichlet_weights'

## LOAD CAMVID ##
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

## COMPILE MODEL ##
if sys.argv[1] == 't':
  weights_name = sys.argv[3]#'baseline_weights'
  T = float(sys.argv[2])
  print("Running Calibration Analysis for Baseline Model with a temperature of {}".format(T))
elif sys.argv[1] == 'd':
  print("Running Calibration Analysis for Dirichlet Model")
  weights_name = sys.argv[2]#'dirichlet_weights'

## LOAD WEIGHTS ##
model.load_weights("./Weights/{}.h5".format(weights_name))

## POP SOFTMAX LAYER ##
if sys.argv[1] == 't':
  model.layers.pop()
  i = model.input
  o = model.layers[-1].output
  model = keras.models.Model(inputs=i, outputs=[o])

## CALIBRATION ANALYSIS ##
files = test_files + val_files
batch_size = 1

sum_bin = np.zeros((20,))
count_right = np.zeros((20,))
total = np.zeros((20,))

files = test_files + val_files
i = 0
for (X,y) in generate_data(files,1,x2y,rgb2label,x_dir,y_dir):

  if sys.argv[1] == 't': # Temperature
    prediction = calibration_softmax(model.predict(X,steps=1)/T)
  elif sys.argv[1] == 'd': # Dirichlet prediction is its expectation
    alphas = model.predict(X,steps=1)
    alpha_0 = np.reshape(np.sum(alphas, axis=3),(alphas.shape[0],alphas.shape[1],alphas.shape[2],1))
    prediction = alphas/alpha_0
    #d = np.array([np.random.dirichlet(1000*s+1e-8) for i in alphas for a in i for s in a])
    #prediction = np.reshape(d, alphas.shape)

  c = np.argmax(y,axis=3)
  mask = (c != 0) + 0 # make void mask
  acc = np.argmax(prediction[mask==1],axis=1) - c[mask==1]
  n_bin = np.max(prediction[mask==1],axis=1)

  # fill bins
  range_min = [0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
  range_max = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]

  for j in range(len(range_min)):
    cropped_pred = n_bin[n_bin < range_max[j]]
    bin_sample = cropped_pred[cropped_pred > range_min[j]]

    cropped_acc = acc[n_bin < range_max[j]]
    bin_acc = cropped_acc[cropped_pred > range_min[j]]

    count_right[j] = count_right[j] + np.count_nonzero(bin_acc==0)
    sum_bin[j] = sum_bin[j] + sum(bin_sample)
    total[j] = total[j] + len(bin_sample)

  i = i+1

  confidence = (sum_bin/(total+1e-12))
  accuracy = (count_right/(total+1e-12))
  gap = np.abs(accuracy - confidence)

  print("Iteration: {}/{}".format(i,len(files)))
  print("acc = {}".format(sum(count_right)/sum(total+1e-12)))
  print(gap)

print("Confidence =", confidence)
print("Accuracy =", accuracy)
print("Gap =", gap)

# ECE
print("ECE = {:.5f}".format(sum((total * gap)/sum(total))))

# MCE
print("MCE = {:.5f}".format(np.max(gap)))
