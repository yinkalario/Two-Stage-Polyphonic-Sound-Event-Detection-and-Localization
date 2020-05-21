import math
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def interpolate(x, ratio):
    '''

    Interpolate the x to have equal time steps as targets

    Input:
        x: (batch_size, time_steps, class_num)
    Output:
        out: (batch_size, time_steps*ratio, class_num) 
    '''
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(ratio, axis=2)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    
    return upsampled

    
def init_layer(layer, nonlinearity='leaky_relu'):
    '''
    Initialize a layer
    '''
    classname = layer.__class__.__name__
    if (classname.find('Conv') != -1) or (classname.find('Linear') != -1):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
        if hasattr(layer, 'bias'):
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(layer.weight, 1.0, 0.02)
        nn.init.constant_(layer.bias, 0.0)


# def init_bn(bn):
#     """Initialize a Batchnorm layer. """
    
#     bn.bias.data.fill_(0.)
#     bn.running_mean.data.fill_(0.)
#     bn.weight.data.fill_(1.)
#     bn.running_var.data.fill_(1.)


def init_gru(rnn):
    """Initialize a GRU layer. """
    
    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)
    
        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in : (i + 1) * fan_in, :])
        
    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
    
    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, 'weight_ih_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

        _concat_init(
            getattr(rnn, 'weight_hh_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, 
                kernel_size=(3,3), stride=(1,1), padding=(1,1)):
        
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()
        
    def init_weights(self):
        
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.bn1)
        init_layer(self.bn2)
        
    def forward(self, x, pool_type='avg', pool_size=(2, 2)):
        
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'frac':
            fractional_maxpool2d = nn.FractionalMaxPool2d(kernel_size=pool_size, output_ratio=1/np.sqrt(2))
            x = fractional_maxpool2d(x)

        return x
