import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utilities import ConvBlock, init_gru, init_layer, interpolate


class VGG9(nn.Module):
    def __init__(self, class_num, pool_type='avg', pool_size=(2,2), interp_ratio=16, pretrained_path=None):
        
        super().__init__()

        self.class_num = class_num
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.interp_ratio = interp_ratio

        self.conv_block1 = ConvBlock(in_channels=10, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.event_fc = nn.Linear(512, class_num, bias=True)
        self.azimuth_fc = nn.Linear(512, class_num, bias=True)
        self.elevation_fc = nn.Linear(512, class_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.event_fc)
        init_layer(self.azimuth_fc)
        init_layer(self.elevation_fc)

    def forward(self, x):
        '''input: (batch_size, mic_channels, time_steps, mel_bins)'''

        x = self.conv_block1(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block2(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block3(x, self.pool_type, pool_size=self.pool_size)
        x = self.conv_block4(x, self.pool_type, pool_size=self.pool_size)
        '''(batch_size, feature_maps, time_steps, mel_bins)'''

        if self.pool_type == 'avg':
            x = torch.mean(x, dim=3)
        elif self.pool_type == 'max':
            (x, _) = torch.max(x, dim=3)
        '''(batch_size, feature_maps, time_steps)'''

        x = x.permute(0, 2, 1)
        '''(batch_size, time_steps, feature_maps)'''

        event_output = torch.sigmoid(self.event_fc(x))
        azimuth_output = self.azimuth_fc(x)
        elevation_output = self.elevation_fc(x)     
        '''(batch_size, time_steps, class_num)'''

        # Interpolate
        event_output = interpolate(event_output, self.interp_ratio)
        azimuth_output = interpolate(azimuth_output, self.interp_ratio) 
        elevation_output = interpolate(elevation_output, self.interp_ratio)
        
        output = {
            'events': event_output,
            'doas': torch.cat((azimuth_output, elevation_output), dim=-1)
        }

        return output


class pretrained_VGG9(VGG9):
    def __init__(self, class_num, pool_type='avg', pool_size=(2,2), interp_ratio=16, pretrained_path=None):

        super().__init__(class_num, pool_type, pool_size, interp_ratio, pretrained_path)

        self.init_weights(pretrained_path)

    def init_weights(self, pretrained_path):

        model = VGG9(self.class_num, self.pool_type, self.pool_size, self.interp_ratio)
        checkpoint = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model_state_dict'])

        self.conv_block1 = model.conv_block1
        self.conv_block2 = model.conv_block2
        self.conv_block3 = model.conv_block3
        self.conv_block4 = model.conv_block4

        init_layer(self.event_fc)
        init_layer(self.azimuth_fc)
        init_layer(self.elevation_fc)

