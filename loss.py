#! /usr/bin/python
# -*- coding: utf8 -*-
import tensorlayer as tl
import tensorflow as tf

def TV_loss(input_tensor):
    tensor_shape = input_tensor.shape.as_list()
    h_tensor = tensor_shape[1]
    w_tensor = tensor_shape[2]
    h_tv_loss = tl.cost.mean_squared_error(input_tensor[:, 1:, :, :], input_tensor[:, :h_tensor-1, :, :], is_mean=True)
    w_tv_loss = tl.cost.mean_squared_error(input_tensor[:, :, 1:, :], input_tensor[:, :, :w_tensor-1, :], is_mean=True)
    tv_loss = h_tv_loss + w_tv_loss
    return  tv_loss
