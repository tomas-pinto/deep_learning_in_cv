import keras
import numpy as np
import matplotlib.pyplot as plt

## GENERATOR ##
# Define the Keras sequence that generates batches for training and validation
class generate_data(keras.utils.Sequence):
    def __init__(self, train_files, batch_size, x2y, rgb2label,x_dir,y_dir):
        # Set batch size
        self.shuffle = True
        self.batch_size = batch_size
        self.n_classes = 12

	# Set mean and std of dataset
        self.mean = [0.41189489566336, 0.4251328133025, 0.4326707089857]
        self.std = [0.27413549931506, 0.28506257482912, 0.28284674400252]

        # Set directories
        self.x_dir = x_dir
        self.y_dir = y_dir

        self.train_files = train_files #filenames of dataset
        self.indexes = np.arange(len(self.train_files)) # indexes of training files

        self.x2y = x2y #dictionary that converts raw to label filenames
        self.rgb2label = rgb2label #dictionary that labels a image into 11 classes

    def __len__(self):
        # Return number of batches that this generator should produce
        return int(np.floor(len(self.train_files) / float(self.batch_size)))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        train_files_batch = [self.train_files[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(train_files_batch)

        return X, y

    def on_epoch_end(self):
        'Shuffles indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, train_files_batch):
        'Generates data containing batch_size samples'

        # Generate raw data and resize by half
        #x_rgb = np.array([cv2.resize(plt.imread(self.x_dir+'/'+img_name),(480,360),interpolation = cv2.INTER_AREA) for img_name in train_files_batch])
        x_rgb = np.array([plt.imread(self.x_dir+'/'+img_name) for img_name in train_files_batch])

        # Generate labeled data and resize by half
        y_train_files_batch = [self.y_dir+'/'+self.x2y[img_name] for img_name in train_files_batch]
        y_rgb = np.array([plt.imread(img_name) for img_name in y_train_files_batch])

	    # Normalize input tensor
        x_rgb = (x_rgb - self.mean)/self.std

        # Initialize label tensor with 0's in it
        y_one_hot = np.zeros((y_rgb.shape[0],y_rgb.shape[1],y_rgb.shape[2],self.n_classes),dtype=np.float64)

        # Fill label tensor
        for batch in range(y_rgb.shape[0]):
            for row in range(y_rgb.shape[1]):
                for col in range(y_rgb.shape[2]):
                    y_one_hot[batch][row][col][self.rgb2label[tuple((y_rgb[batch][row][col]*255).reshape(1, -1)[0])]] = 1

        return x_rgb, y_one_hot
