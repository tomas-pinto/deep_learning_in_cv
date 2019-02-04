# Deep Learning in Computer Vision Final Project

The objective was to compare the performance of Dirichlet output layers and Temperature Scaling in the calibration of Mobilenet V2 on CamVid dataset.

## Running train_model.py:
- python train_model.py <type_of_training> <L2 reg> <dropout rate> <loss> <weights (optional)>
  - <type_of_training> must be 'data_augmentation' or 'finetuning'
  - <loss> must be 'dirichlet' or 'crossentropy'
  - <weights> must be the name of the weight file and be in Weights folder
