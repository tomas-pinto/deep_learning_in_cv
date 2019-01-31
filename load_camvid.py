import os
import numpy as np

## MAKE SETS AND DICTIONARIES ##

def make_x2y(x_directory, y_directory):
    # Create a sorted list of input and labeled data filenames
    x_file_list = sorted(os.listdir(x_directory))
    y_file_list = sorted(os.listdir(y_directory))

    # Create a Dictionary of filenames that maps input x to labels y
    x2y = dict(zip(x_file_list,y_file_list))

    return x2y

def make_rgb2label(label_directory):
    # Create a Dictionary to convert rgb data to labeled data

    f = open(label_directory,"r")

    # Group all classes into 12 different classes (11 channels + 1 mask)
    rgb2label = {}
    # Grouping indexes
    values = [10,2,11,2,2,9,10,10,3,8,4,4,7,11,9,5,10,4,5,5,7,1,9,3,7,6,9,2,6,0,2]
    index = 0
    for x in f:
        line = x.split('\t')[0]
        result = line.split(' ')
        arr = np.array(result).astype(np.float)
        key = tuple(arr.reshape(1, -1)[0])
        rgb2label[key] = values[index]
        index = index + 1
    f.close()

    return rgb2label

# Get the training, validation and test sets
def make_set(filename):
    with open(filename) as f:
        set_files = f.read().splitlines()
    f.close()

    return set_files
