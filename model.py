import time
import tensorlayer as tl
import tensorflow as tf
from tensorlayer.layers import *
from layers import *


def CinCGAN_g(t_image, strides, is_train=False, reuse=False, name='generator'):
    w_init = tf.random_normal_initializer(stddev=0.02)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope(name, reuse=reuse):
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (7, 7), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init)
        n = Conv2d(n, 64, (3, 3), (strides, strides), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init)
        n = Conv2d(n, 64, (3, 3), (strides, strides), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init)

        for i in range(6):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init)
            nn = BatchNormLayer(nn, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=g_init, name='k3n64s1/b1/%s' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init)
            nn = BatchNormLayer(nn, act=tf.nn.leaky_relu, is_train=is_train, gamma_init=g_init, name='k3n64s1/b2/%s' % i)
            #nn = Residual_scale(nn, 0.1)
            nn = ElementwiseLayer([n, nn], tf.add)
            n = nn
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init)
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init)
        n = Conv2d(n, 3, (7, 7), (1, 1), act=None, padding='SAME', W_init=w_init)

        return n

def CinCGAN_d(t_image, strides, is_train=False, reuse=False, name='discriminator'):
    w_init = tf.random_normal_initializer(stddev=0.02)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope(name, reuse=reuse):
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (4, 4), (strides, strides), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init)

        n = Conv2d(n, 128, (4, 4), (strides, strides), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init)
        n = BatchNormLayer(n, act=None, is_train=is_train, gamma_init=g_init, name='k4n128s1/b')

        n = Conv2d(n, 256, (4, 4), (strides, strides), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init)
        n = BatchNormLayer(n, act=None, is_train=is_train, gamma_init=g_init, name='k4n256s1/b')

        n = Conv2d(n, 512, (4, 4), (1, 1), act=tf.nn.leaky_relu, padding='SAME', W_init=w_init)
        n = BatchNormLayer(n, act=None, is_train=is_train, gamma_init=g_init, name='k4n512s1/b')

        n = Conv2d(n, 1, (4, 4), (1, 1), act=None, padding='SAME', W_init=w_init)

        return n

