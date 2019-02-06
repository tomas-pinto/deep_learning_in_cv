import numpy as np
from scipy.optimize import minimize
from Utils.model import calibration_softmax

class TemperatureScaling():

    def __init__(self, model, temp = 1., maxiter = 50, solver = "L-BFGS-B"):
        """
        Initialize class

        Params:
            temp (float): starting temperature, default 1
            maxiter (int): maximum iterations done by optimizer, however 8 iterations have been maximum.
        """
        self.temp = temp
        self.maxiter = maxiter
        self.solver = solver
        self.model = model
        self.i = 0

    def _loss_fun(self, x, generator):
        sum_bin = np.zeros((10,))
        count_right = np.zeros((10,))
        total = np.zeros((10,))

        X,y = generator.__getitem__(self.i)
        prediction = calibration_softmax(self.model.predict(X,steps=1)/x)

        c = np.argmax(y,axis=3)
        mask = (c != 0) + 0 # make void mask
        acc = np.argmax(prediction[mask==1],axis=1) - c[mask==1]
        n_bin = np.max(prediction[mask==1],axis=1)

        # fill bins
        range_min = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        range_max = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

        for j in range(len(range_min)):
           cropped_pred = n_bin[n_bin < range_max[j]]
           bin_sample = cropped_pred[cropped_pred > range_min[j]]

           cropped_acc = acc[n_bin < range_max[j]]
           bin_acc = cropped_acc[cropped_pred > range_min[j]]

           count_right[j] = count_right[j] + np.count_nonzero(bin_acc==0)
           sum_bin[j] = sum_bin[j] + sum(bin_sample)
           total[j] = total[j] + len(bin_sample)

        self.i = self.i + 1

        confidence = (sum_bin/(total+1e-12))
        accuracy = (count_right/(total+1e-12))
        gap = np.abs(accuracy - confidence)

        # ECE
        ece = sum((total * gap)/sum(total))
        print("Temp: ", x, " ECE: ", ece)

        return ece

    # Find the temperature
    def fit(self, generator):
        """
        Trains the model and finds optimal temperature

        Params:
            logits: the output from neural network for each class (shape [samples, classes])
            true: one-hot-encoding of true labels.

        Returns:
            the results of optimizer after minimizing is finished.
        """

        #true = true.flatten() # Flatten y_val
        opt = minimize(self._loss_fun, x0 = 0.1, args=(generator), options={'maxiter':self.maxiter}, method = self.solver)
        self.temp = opt.x[0]

        return opt
