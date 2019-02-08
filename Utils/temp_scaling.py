import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from Utils.model import calibration_softmax

def _calculate_calibration(prediction,y):
    sum_bin = np.zeros((10,))
    count_right = np.zeros((10,))
    total = np.zeros((10,))

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

    confidence = (sum_bin/(total+1e-12))
    accuracy = (count_right/(total+1e-12))
    gap = np.abs(accuracy - confidence)

    # ECE
    ece = sum((total * gap)/sum(total))
    mce = np.max(gap)

    return ece,mce

class TemperatureScaling():

    def __init__(self, temp = 1., tol = 0.001, solver = "L-BFGS-B"):
        """
        Initialize class
        Params:
            temp (float): starting temperature, default 1
            maxiter (int): maximum iterations done by optimizer, however 8 iterations have been maximum.
        """
        self.temp = temp
        self.tol = tol
        self.solver = solver

    def _loss_fun(self, x, probs, true, loss):
        prediction = self.predict(probs, x)

        if loss == 'nll':
            c = np.argmax(true,axis=1)
            mask = (c != 0) + 0 # make void mask
            #chosen_loss = np.sum(-np.log(np.sum(true[mask==1] * prediction[mask==1], axis=1)))
            chosen_loss = log_loss(y_true=true[mask==1], y_pred=prediction[mask==1])
            print("Temp: ", x, " NLL: ", chosen_loss)

        else:
            ece, mce = _calculate_calibration(prediction,true)

            if loss == 'ece':
                print("Temp: ", x, " ECE: ", ece)
                chosen_loss = ece
            elif loss == 'mce':
                print("Temp: ", x, " MCE: ", mce)
                chosen_loss = mce

        return chosen_loss

    # Find the temperature
    def fit(self, logits, true, loss):
        """
        Trains the model and finds optimal temperature
        Params:
            logits: the output from neural network for each class (shape [samples, classes])
            true: one-hot-encoding of true labels.
        Returns:
            the results of optimizer after minimizing is finished.
        """

        #true = true.flatten() # Flatten y_val
        opt = minimize(self._loss_fun, x0 = 0.1, args=(logits, true, loss), tol=self.tol, method = self.solver)
        self.temp = opt.x[0]

        return opt

    def predict(self, logits, temp = None):
        """
        Scales logits based on the temperature and returns calibrated probabilities
        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            temp: if not set use temperatures find by model or previously set.
        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """

        if not temp:
            return calibration_softmax(logits/self.temp)
        else:
            return calibration_softmax(logits/temp)
