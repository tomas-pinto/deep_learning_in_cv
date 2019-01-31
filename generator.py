import matplotlib.pyplot as plt
import numpy as np

def generate_data(train_files,batch_size,x2y,rgb2label,x_dir,y_dir):
    i = 0
    while (i != len(train_files)):
        x_batch = []
        y_batch = []
        for b in range(batch_size):
            # Read label and input images from files
            x_rgb = np.array([plt.imread(x_dir+'/'+img_name) for img_name in train_files[i:(i+1)]])
            y_rgb = np.array([plt.imread(y_dir+'/'+x2y[train_files[i:(i+1)][0]])])

            # Append images read to the batch
            x_batch.append(x_rgb[0])
            y_batch.append(y_rgb[0])

            i += 1

        # Convert images batches to numpy arrays
        x_batch_np = np.array(x_batch)
        y_batch_np = np.array(y_batch)

        # Initialize label tensor with -1 in it
        y_one_hot = np.zeros((y_batch_np.shape[0],y_batch_np.shape[1],y_batch_np.shape[2],1))

        # Fill label tensor
        for batch in range(y_batch_np.shape[0]):
          for row in range(y_batch_np.shape[1]):
            for col in range(y_batch_np.shape[2]):
              y_one_hot[batch][row][col][rgb2label[tuple((y_rgb[batch][row][col]*255).reshape(1, -1)[0])]] = 1

        yield (x_batch_np, y_one_hot)
