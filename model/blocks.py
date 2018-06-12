import torch
import torch.nn as nn
import torch.nn.functional as F

class double_conv_relu(nn.Module):
    
    def __init__(self, in_channels, out_channels, dropout=False):
        super(double_conv_relu, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.drop = nn.Dropout2d(p=0.2)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.ReLU = nn.ReLU(inplace=True)
        self.dropout = dropout
    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.ReLU(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.ReLU(out)
        if(self.dropout):
            out = self.drop(out)
        return out
    


class upsample(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(upsample, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
    def forward(self, x):
        out = self.up(x)
        return out
    
class concatenate_conv(nn.Module):
    def __init__(self, layer_size):
        super(concatenate_conv, self).__init__()
        self.conv = double_conv_relu(layer_size*2, layer_size)
        
    def forward(self, encoder_layer, decoder_layer):
        out = torch.cat([encoder_layer, decoder_layer], dim=1)
        out = self.conv(out)
        return out