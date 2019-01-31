## MODEL DEFINITION ##
# -*- coding: utf-8 -*-
import numpy as np

from keras.models import Model

from keras import layers
from keras import regularizers

from keras.layers import Input, Activation, Concatenate, Add, Dropout, BatchNormalization
from keras.layers import Conv2D, DepthwiseConv2D, ZeroPadding2D, AveragePooling2D, UpSampling2D
from keras.engine import Layer
from keras.engine import InputSpec

from keras.engine.topology import get_source_inputs
from keras import backend as K
from keras.applications import imagenet_utils
from keras.utils import conv_utils
from keras.utils.data_utils import get_file

def calibration_softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=1)


def relu6(x):
    return K.relu(x, max_value=6)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id,
                        skip_connection, rate=1, reg=0.0001):

    in_channels = inputs._keras_shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'expanded_conv_{}_'.format(block_id)

    # Expand
    x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
               use_bias=False, activation=None, kernel_regularizer=regularizers.l2(reg),
               name=prefix + 'expand')(x)
    x = BatchNormalization(name=prefix + 'expand_BN')(x)
    x = Activation(relu6, name=prefix + 'expand_relu')(x)

    # Depthwise
    x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
                        use_bias=False, padding='same', dilation_rate=(rate, rate),
                        depthwise_regularizer=regularizers.l2(reg), name=prefix + 'depthwise')(x)
    x = BatchNormalization(name=prefix + 'depthwise_BN')(x)

    x = Activation(relu6, name=prefix + 'depthwise_relu')(x)

    # Project
    x = Conv2D(pointwise_filters,
               kernel_size=1, padding='same', use_bias=False, activation=None,
               kernel_regularizer=regularizers.l2(reg), name=prefix + 'project')(x)
    x = BatchNormalization(name=prefix + 'project_BN')(x)

    if skip_connection:
        return Add(name=prefix + 'add')([inputs, x])
    return x


def mobilenetV2(input_shape=(512, 512, 3), classes=21, alpha=1., reg=0.0001, d=0.1):

    img_input = Input(shape=input_shape)

    # Input Layer
    OS = 8
    first_block_filters = _make_divisible(32 * alpha, 8)
    x = Conv2D(first_block_filters,
               kernel_size=3, strides=(2, 2), padding='same',
               use_bias=False, kernel_regularizer=regularizers.l2(reg), name='Conv')(img_input)
    x = BatchNormalization(name='Conv_BN')(x)
    x = Activation(relu6, name='Conv_Relu6')(x)

    # Inverted Residual Blocks
    x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                            expansion=1, block_id=0, skip_connection=False)

    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                            expansion=6, block_id=1, skip_connection=False)
    x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=2, skip_connection=True)

    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                            expansion=6, block_id=3, skip_connection=False)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=4, skip_connection=True)
    x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=5, skip_connection=True)

    # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,  # 1!
                            expansion=6, block_id=6, skip_connection=False)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=7, skip_connection=True)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=8, skip_connection=True)
    x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=9, skip_connection=True)

    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=10, skip_connection=False)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=11, skip_connection=True)
    x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                            expansion=6, block_id=12, skip_connection=True)

    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=2,  # 1!
                            expansion=6, block_id=13, skip_connection=False)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=14, skip_connection=True)
    x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=15, skip_connection=True)

    x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, rate=4,
                            expansion=6, block_id=16, skip_connection=False)

    # end of feature extractor

    # branching for Atrous Spatial Pyramid Pooling

    # Image Feature branch
    b4 = AveragePooling2D(pool_size=(int(np.ceil(input_shape[0] / OS)), int(np.ceil(input_shape[1] / OS))))(x)
    b4 = Conv2D(256, (1, 1), padding='same', kernel_regularizer=regularizers.l2(reg),
                use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_BN')(b4)
    b4 = Activation('relu')(b4)
    b4 = UpSampling2D(size=(int(np.ceil(input_shape[0] / OS)), int(np.ceil(input_shape[1] / OS))), interpolation='bilinear')(b4)

    # simple 1x1
    b0 = Conv2D(256, (1, 1), padding='same', kernel_regularizer=regularizers.l2(reg), use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_BN')(b0)
    b0 = Activation('relu', name='aspp0_activation')(b0)

    # there are only 2 branches in mobilenetV2. not sure why
    x = Concatenate()([b4, b0])

    x = Conv2D(256, (1, 1), padding='same', kernel_regularizer=regularizers.l2(reg),
               use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_BN')(x)
    x = Activation('relu')(x)
    x = Dropout(d)(x)

    # DeepLab v.3+ decoder
    # you can use it with arbitary number of classes
    last_layer_name = 'last_layer'

    x = Conv2D(classes, (1, 1), padding='same', kernel_regularizer=regularizers.l2(reg),
               name=last_layer_name)(x)
    x = UpSampling2D(size=(8, 8), interpolation='bilinear')(x)
    x = Conv2D(classes, (1, 1), padding='same',
               kernel_regularizer=regularizers.l2(reg), name='conversion1hot')(x)
    x = Activation('softmax')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    model = Model(img_input, x, name='mobilenet_v2')

    return model
