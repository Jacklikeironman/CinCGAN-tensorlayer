#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
from tensorlayer.layers.core import Layer
from tensorlayer import logging

class Residual_scale(Layer):
    def __init__(
            self,
            prev_layer,
            residual_scale = 0.1,
            name = 'Residual_scale_layer'):
        super(Residual_scale, self).__init__(prev_layer=prev_layer, name=name)

        logging.info("Residual_scale_layer %s: Residual_scale: %d" % (self.name, residual_scale))

        self.outputs = self.inputs * 0.1
        self._add_layers(self.outputs)

class ReLu(Layer):
    def __init__(
            self,
            prev_layer,
            name = 'ReLu_layer'):
        super(ReLu, self).__init__(prev_layer=prev_layer, name=name)

        logging.info("ReLu_layer %s" % (self.name))

        self.outputs = tf.nn.relu(self.inputs)
        self._add_layers(self.outputs)