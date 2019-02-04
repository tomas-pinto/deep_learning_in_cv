## CUSTOM LOSS FUNCTIONS ##
from keras import backend as K

# Weighted crossentropy loss #
def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss

# Weighted Dirichlet Loss #
def weighted_dirichlet_loss(weights):

    class_weights = K.constant(weights)

    def loss(y_true, y_pred):
        pixel_weights = K.gather(class_weights, K.argmax(y_true, axis=-1))
        dist = tf.distributions.Dirichlet(1000*y_pred+K.epsilon())
        error = -dist.log_prob(y_true)
        loss = tf.reduce_sum(pixel_weights*error)

        return loss

    return loss
