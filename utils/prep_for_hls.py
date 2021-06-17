#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 21:17:54 2021

@author: abdel
"""

import os
import numpy as np


import hls4ml
from hls4ml.utils.config import create_vivado_config
from hls4ml.converters import _check_hls_config, _check_model_config
from hls4ml.converters.pytorch_to_hls import PyTorchModelReader

# locals 
from .layer_handlers import layer_handlers

def config_from_torch_model(model, name, input_shape):
    hls_config = hls4ml.utils.config_from_pytorch_model(model)
    
    config = create_vivado_config(output_dir = os.path.join(os.getcwd(), "hls_output", name))
    config['PytorchModel'] = model
    config['InputShape'] = input_shape
    model_config = hls_config.get('Model', None)
    config['HLSConfig']['Model'] = _check_model_config(model_config)
    _check_hls_config(config, hls_config)
    return config
    
def prep_for_hls(model, name, input_shape):

    config = config_from_torch_model(model, name, input_shape)
    
    layer_list = []
    reader = PyTorchModelReader(config)
    input_shapes = [list(reader.input_shape)]
    model = reader.torch_model
        
    skip_layers = ['Dropout', 'Flatten', 'Sequential']
    supported_layers = ['Conv1d', 'Conv2d', 'Linear', 'LeakyReLU', 'ThresholdedReLU', 'ELU', 'PReLU', 'Softmax', 'ReLU', 'BatchNorm2d', 'BatchNorm1d', 'Dropout', 'Flatten', 'Sequential']
    
    output_shapes = {}
    output_shape = None
    layer_counter = 0
    input_layer = {}
    input_layer['name'] = 'input1'
    input_layer['class_name'] = 'InputLayer'
    input_layer['input_shape'] = input_shapes[0][1:]
    layer_list.insert(0, input_layer)
        
    for layer_name, pytorch_layer in model.named_modules():
        
        pytorch_class = pytorch_layer.__class__.__name__
        
        #First module is the whole model's class
        if pytorch_class == model.__class__.__name__:
            continue
        
        if pytorch_class not in supported_layers:
            raise Exception('Unsupported layer {}'.format(pytorch_class))
                
        #If not the first layer then input shape is taken from last layer's output
        if layer_counter != 0:
            input_shapes = [output_shape] #In case there are multiple inputs
        
        #Handle skipped layers
        if pytorch_class in skip_layers:
            if pytorch_class == 'Sequential': #Ignore the mother module's class name
                continue

            if pytorch_class == 'Flatten':
                output_shapes[layer_name] = [input_shapes[0][0], np.prod(input_shapes[0][1:])]
            else:
                output_shapes[layer_name] = input_shapes[0]
                
            continue #!!
        
        #Increment the layer counter after initial screenings
        if pytorch_class in supported_layers:
            layer_counter += 1
        
        #Process the layer
        layer, output_shape = layer_handlers[pytorch_class](pytorch_layer, layer_name, input_shapes, reader, config)

        #print('Layer name: {}, layer type: {}, input shape: {}'.format(layer['name'], layer['class_name'], input_shapes))
        layer_list.append(layer)
        
        assert(output_shape is not None)
        output_shapes[layer['name']] = output_shape
            
    return config, reader, layer_list
    


    
    
