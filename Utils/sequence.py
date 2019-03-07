import keras
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2



# DATA AUGMENTATION #
# Randomly crop the image to a specific size. For data augmentation
def _random_crop(image, label, crop_height, crop_width):
    if (image.shape[0] != label.shape[0]) or (image.shape[1] != label.shape[1]):
        raise Exception('Image and label must have the same dimensions!')

    if (crop_width <= image.shape[1]) and (crop_height <= image.shape[0]):
        x = random.randint(0, image.shape[1]-crop_width)
        y = random.randint(0, image.shape[0]-crop_height)

        if len(label.shape) == 3:
            return image[y:y+crop_height, x:x+crop_width, :], label[y:y+crop_height, x:x+crop_width, :]
        else:
            return image[y:y+crop_height, x:x+crop_width, :], label[y:y+crop_height, x:x+crop_width]
    else:
        raise Exception('Crop shape exceeds image dimensions!')

def _data_augmentation(input_image, output_image):
    # Data augmentation
    input_image, output_image = _random_crop(input_image, output_image, 240, 240)
    if random.randint(0,1):
        input_image  = cv2.flip(input_image, 1)
        output_image = cv2.flip(output_image, 1)

    return input_image, output_image


## GENERATOR ##
# Define the Keras sequence that generates batches for training and validation
class generate_data(keras.utils.Sequence):
	def __init__(self, train_files, batch_size,
				x2y, rgb2label, x_dir, y_dir, dirichlet=False, data_aug=False,val_data=True):

		# Set batch size
		self.shuffle = True
		self.batch_size = batch_size
		self.n_classes = 12
		self.data_augmentation = data_aug
		self.dirichlet = dirichlet

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

		#################### ADDED BY MORITZ #####################################
		self.epoch_count = 1
		self.val_data = val_data
###########################################################################

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
#################### ADDED BY MORITZ #####################################          
		self.epoch_count += 1
#####################################################################

	def __data_generation(self, train_files_batch):
		'Generates data containing batch_size samples'

		# Generate raw data and resize by half
		#x_rgb = np.array([cv2.resize(plt.imread(self.x_dir+'/'+img_name),(480,360),interpolation = cv2.INTER_AREA) for img_name in train_files_batch])
		x_rgb = np.array([plt.imread(self.x_dir+'/'+img_name) for img_name in train_files_batch])

		# Generate labeled data and resize by half
		y_train_files_batch = [self.y_dir+'/'+self.x2y[img_name] for img_name in train_files_batch]
		y_rgb = np.array([plt.imread(img_name) for img_name in y_train_files_batch])

		if self.data_augmentation == True:
			x = np.empty([y_rgb.shape[0]*12, 240, 240, 3])
			y = np.empty([y_rgb.shape[0]*12, 240, 240, 3])

			#Data Augmentation
			i = 0
			for b in range(y_rgb.shape[0]):
				for _ in range(12):
					z,w = _data_augmentation(x_rgb[b], y_rgb[b])
					x[i,:,:,:] = z*255
					y[i,:,:,:] = w*255
					i += 1

			x_rgb = x/255
			y_rgb = y/255

		# Normalize input tensor
		x_rgb = (x_rgb - self.mean)/self.std

		# Initialize label tensor with 0's in it
		y_one_hot = np.zeros((y_rgb.shape[0],y_rgb.shape[1],y_rgb.shape[2],self.n_classes),dtype=np.float64)

		# Fill label tensor
		for batch in range(y_rgb.shape[0]):
			for row in range(y_rgb.shape[1]):
				for col in range(y_rgb.shape[2]):
					y_one_hot[batch][row][col][self.rgb2label[tuple((y_rgb[batch][row][col]*255).reshape(1, -1)[0])]] = 1
			
		print(self.dirichlet)			
		if self.dirichlet == True:
			# Laplacian Smoothing
			wished_lapla_p = 0.000001
			####################### ADDED BY MORITZ ##############################
			if self.val_data == False:
				#procent of the epochs after which the parameter no longer changes
				lapla_conv_p = 0.5
				max_epoch = 150
				#the wished start for the lapla_smo_p
				wished_lapla_start = 0.3
				#linear annealing
				lap_smo_par = wished_lapla_start - ((self.epoch_count*(wished_lapla_start-wished_lapla_p))/(max_epoch*lapla_conv_p))
				#quadratic annealing
					#coming

				#ensures laplace is not smaller than thought
				if(self.epoch_count >= (max_epoch * lapla_conv_p) ):
					lap_smo_par = wished_lapla_p

				print("laplace smoothing:")
				print(lap_smo_par)
				#Laplace Smoothing
				temp = 1 + lap_smo_par * self.n_classes
				y_one_hot = (y_one_hot + lap_smo_par)/temp
			else:
				print("laplace smoothing without annealing:")
				print(wished_lapla_p)
				temp = 1 + wished_lapla_p * self.n_classes
				y_one_hot = (y_one_hot + wished_lapla_p)/temp

		



	#####################################################################
		return x_rgb, y_one_hot
		
