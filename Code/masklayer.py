from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf

class MaskLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MaskLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.bias = self.add_weight(name='bias',
                                    shape=(self.output_dim),
                                    initializer='he_normal',
                                    trainable=True)
        super(MaskLayer, self).build(input_shape) 

    def call(self, inputs):

        outputs = K.bias_add(
            K.zeros(K.shape(inputs)),
            self.bias)
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)