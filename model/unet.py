from blocks import *
from torch.autograd import Variable
from torch import optim
import torchvision
from torch.utils.data import DataLoader
from skimage import io
import numpy as np
from skimage import io
import os


        

class unet(nn.Module):
    def __init__(self, in_channels, out_classes, dropout=False):
        super(unet, self).__init__()
        
        self.encoder_conv1 = double_conv_relu(in_channels, 64, dropout)
        self.encoder_conv2 = double_conv_relu(64, 128, dropout)
        self.encoder_conv3 = double_conv_relu(128, 256, dropout)
        self.encoder_conv4 = double_conv_relu(256, 512, dropout)
        self.encoder_conv5 = double_conv_relu(512, 1024, dropout) #set out channels to 512 instead of 1024 for memory
        
        self.decoder_conv1 = concatenate_conv(512)
        self.decoder_conv2 = concatenate_conv(256)
        self.decoder_conv3 = concatenate_conv(128)
        self.decoder_conv4 = concatenate_conv(64)
        
        self.up1 = upsample(1024, 512)
        self.up2 = upsample(512, 256)
        self.up3 = upsample(256, 128)
        self.up4 = upsample(128, 64)
        
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.output_conv = nn.Conv2d(64, out_classes, kernel_size=1)
        
    def forward(self, x):
        encode1 = self.encoder_conv1(x)
        out = self.down(encode1)
        encode2 = self.encoder_conv2(out)
        out = self.down(encode2)
        encode3 = self.encoder_conv3(out)
        out = self.down(encode3)
        encode4 = self.encoder_conv4(out)
        out = self.down(encode4)
        encode5 = self.encoder_conv5(out)
        decode = self.up1(encode5)
        decode = self.decoder_conv1(encode4, decode)
        decode = self.up2(decode)
        decode = self.decoder_conv2(encode3, decode)
        decode = self.up3(decode)
        decode = self.decoder_conv3(encode2, decode)
        decode = self.up4(decode)
        decode = self.decoder_conv4(encode1, decode)
        out = self.output_conv(decode)
        
        return out
        
# if __name__ == '__main__':
# 	model = unet(1, 2)
# 	for i in range(1):
# 		x = Variable(torch.FloatTensor(np.random.random((1, 1, 256, 256))))
# 		out = model(x)
# 		loss = torch.sum(out)
# 		loss.backward()
# 		print(loss)