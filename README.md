# Deep Learning in Computer Vision Final Project

The objective was to compare the performance of Dirichlet output layers and Temperature Scaling in the calibration of Mobilenet V2 on CamVid dataset.

## Training a model:
- Type in Command Line: python train_model.py [type_of_training] [L2 reg] [dropout rate] [loss] [weights (optional)]
  - [type_of_training] must be 'data_augmentation' or 'finetuning'
  - [loss] must be 'dirichlet' or 'crossentropy'
  - [weights] must be the name of the weight file and be in Weights folder

## Calibration Analysis of a model:
- Type in Command Line: python calibration_analysis.py [type_of_model] [desired_T] [weights]
  - [type_of_model] must be 'd' for dirichlet model or 't' for baseline model with temperature scaling
  - [desired_T] must be a number bigger than zero (this option is only available for baseline model)
  - [weights] must be the name of the weight file and be in Weights folder

## Temperature Calculation for a model:
- Type in Command Line: python calibration_analysis.py [type_of_model] [desired_T] [weights]
  - [type_of_model] must be 'd' for dirichlet model or 't' for baseline model with temperature scaling
  - [desired_T] must be a number bigger than zero (this option is only available for baseline model)
  - [weights] must be the name of the weight file and be in Weights folder
