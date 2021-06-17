#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 23:34:02 2021

@author: abdel
"""

import math
from hls4ml.converters.pytorch_to_hls import pytorch_handler
from hls4ml.converters.utils import *
layer_handlers = {}

#@pytorch_handler('Conv1d')
def parse_conv1d_layer(pytorch_layer, layer_name, input_shapes, data_reader, config):
    assert('Conv1d' in pytorch_layer.__class__.__name__)
    
    layer = {}
    
    layer['name'] = layer_name
    layer['class_name'] = 'Conv1D'
    layer['data_format'] = 'channels_first' #Pytorch default (can't change)
    
    #Input info
    (
        layer['in_width'],
        layer['n_chan']
    ) = parse_data_format(input_shapes[0], 'channels_first') #Keras's default is channels_last
    
    #Additional parameters
    layer['n_filt'] = pytorch_layer.out_channels
    layer['filt_width'] = pytorch_layer.kernel_size[0] 
    layer['stride_width'] = pytorch_layer.stride[0]
    layer['pad_left'] = layer['pad_right'] = pytorch_layer.padding[0]
    layer['dilation'] = pytorch_layer.dilation[0]
    
    if pytorch_layer.padding[0] == 0: # No padding, i.e., 'VALID' padding in Keras/Tensorflow
        layer['padding'] = 'valid'
    else: #Only 'valid' and 'same' padding are available in Keras
        layer['padding'] = 'same'
    
    #Ouput info
    (layer['out_width'],_,_) = compute_padding_1d(layer['padding'],
                                                  layer['in_width'],
                                                  layer['stride_width'],
                                                  layer['filt_width'])
    
    output_shape=[input_shapes[0][0], layer['n_filt'], layer['out_width']] #Channel first as default
    
    return layer, output_shape
layer_handlers['Conv1d'] = parse_conv1d_layer

#@pytorch_handler('Conv2d')
def parse_conv2d_layer(pytorch_layer, layer_name, input_shapes, data_reader, config):
    assert('Conv2d' in pytorch_layer.__class__.__name__)
    
    layer = {}
    
    layer['name'] = layer_name
    layer['class_name'] = 'Conv2D'
    layer['data_format'] = 'channels_first' #Pytorch default (can't change)
    
    #Input info
    (
        layer['in_height'],
        layer['in_width'],
        layer['n_chan']
    ) = parse_data_format(input_shapes[0], 'channels_first') #Keras's default is channels_last
    
    #Additional parameters
    layer['n_filt'] = pytorch_layer.out_channels
    layer['filt_height'] = pytorch_layer.kernel_size[0]
    layer['filt_width'] = pytorch_layer.kernel_size[1]
    layer['stride_height'] = pytorch_layer.stride[0]
    layer['stride_width'] = pytorch_layer.stride[1]
    layer['dilation'] = pytorch_layer.dilation[0]
    layer['pad_top'] = layer['pad_bottom'] = pytorch_layer.padding[0]
    layer['pad_left'] = layer['pad_right'] = pytorch_layer.padding[1]
    
    if all(x == 0 for x in pytorch_layer.padding): # No padding, i.e., 'VALID' padding in Keras/Tensorflow
        layer['padding'] = 'valid'
    else: #Only 'valid' and 'same' padding are available in Keras
        layer['padding'] = 'same'
    
    #Ouput info
    (layer['out_height'], layer['out_width'],_,_,_,_) = compute_padding_2d(layer['padding'],
                                                                           layer['in_height'],
                                                                           layer['in_width'],
                                                                           layer['stride_height'],
                                                                           layer['stride_width'],
                                                                           layer['filt_height'],
                                                                           layer['filt_width'])
    
    output_shape = [input_shapes[0][0], layer['n_filt'], layer['out_height'], layer['out_width']]
    
    return layer, output_shape
layer_handlers['Conv2d'] = parse_conv2d_layer

#@pytorch_handler('Linear')
def parse_linear_layer(pytorch_layer, layer_name, input_shapes, data_reader, config):
    assert('Linear' in pytorch_layer.__class__.__name__)
    
    layer = {}
   
    layer['class_name'] = 'Dense'
    layer['name'] = layer_name
    
    layer['n_in'] = pytorch_layer.in_features
    layer['n_out'] = pytorch_layer.out_features
    
    #Handling whether bias is used or not
    if pytorch_layer.bias is None:    
        layer['use_bias'] = False
    else:
        layer['use_bias'] = True
        
    output_shape = [input_shapes[0][0], layer['n_out']]
    
    return layer, output_shape
layer_handlers['Linear'] = parse_linear_layer

activation_layers = ['LeakyReLU', 'ThresholdedReLU', 'ELU', 'PReLU', 'Softmax', 'ReLU']
#@pytorch_handler(*activation_layers)
def parse_activation_layer(pytorch_layer, layer_name, input_shapes, data_reader, config):
    
    layer = {}
    
    layer['class_name'] =  pytorch_layer.__class__.__name__
    layer['activation'] = layer['class_name']
    layer['name'] = layer_name
    
    if layer['class_name'] == 'ReLU':
        layer['class_name'] = 'Activation'
    
    output_shape=input_shapes[0]
    
    return layer, output_shape

for v in activation_layers:
    layer_handlers[v] = parse_activation_layer

batchnorm_layers = ['BatchNorm2d', 'BatchNorm1d']
#@pytorch_handler(*batchnorm_layers)
def parse_batchnorm_layer(pytorch_layer, layer_name, input_shapes, data_reader, config):
    assert('BatchNorm' in pytorch_layer.__class__.__name__)
    
    layer = {}
   
    layer['class_name'] = 'BatchNormalization'
    layer['data_format'] = 'channels_first'
    layer['name'] = layer_name
    
    #batchnorm para
    layer['epsilon'] = pytorch_layer.eps
    
    in_size = 1
    for dim in input_shapes[0][1:]:
        in_size *= dim
        
    layer['n_in'] = layer['n_out'] = in_size
    
    if len(input_shapes[0]) == 2:
        layer['n_filt'] = -1
    elif len(input_shapes[0]) > 2:
        layer['n_filt']=input_shapes[0][1] #Always channel first for Pytorch

    return layer, [shape for shape in input_shapes[0]]

for v in batchnorm_layers:
    layer_handlers[v] = parse_batchnorm_layer

