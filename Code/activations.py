# -*- coding: utf-8 -*-
from __future__ import absolute_import
import keras.backend as K
import tensorflow as tf


def round_through(x):

    rounded = K.round(x)
    return x + K.stop_gradient(rounded - x)


def _hard_sigmoid(x):

    x = (0.5 * x) + 0.5 

    return K.clip(x, 0, 1)


    
def binary_sigmoid(x):

    return round_through(_hard_sigmoid(x))


def step_func(x):
    x = x
    return round_through(K.clip(x,0,1)) 