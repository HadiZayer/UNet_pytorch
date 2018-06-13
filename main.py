import os
from skimage import io
from model.unet import unet
import torchvision
import torch
import train
from model.unet import unet
from torch.autograd import Variable

data_dir = os.path.join((os.getcwd()), 'data')
labels = io.imread(os.path.join(data_dir, 'train-labels.tif')) #load training labels
labels = torchvision.transforms.ToTensor()(labels)
labels.requires_grad = False
labels = labels.transpose(0,1) #needed because of the TIF files

labels = torch.Tensor.long(labels)

imgs = io.imread(os.path.join(data_dir, 'train-volume.tif')) #load training data
imgs = torchvision.transforms.ToTensor()(imgs)
imgs = imgs.transpose(0,1)
imgs.requires_grad = False
imgs = imgs.unsqueeze(1)

test = io.imread(os.path.join(data_dir, 'test-volume.tif')) #load training data
test = torchvision.transforms.ToTensor()(test)
test = test.transpose(0,1)
test.requires_grad = False
test = test.unsqueeze(1)

model = unet(1,2)
model.load_state_dict(torch.load('saved_model_state.pt'))

train.train(model, imgs, labels, batch_size=1, epochs=1)
torch.save(model.state_dict(), 'saved_model_state.pt')

testimg = test[20]
testimg = testimg.unsqueeze(0)
testimg = Variable(testimg)
out = model(testimg)
_, indices = torch.max(out, 1)
torchvision.utils.save_image(test[20], 'test20.png')
torchvision.utils.save_image(indices, 'outtest20.png')

testimg = imgs[20]
testimg = testimg.unsqueeze(0)
testimg = Variable(testimg)
out = model(testimg)
_, indices = torch.max(out, 1)

torchvision.utils.save_image(testimg, 'img20.png')
torchvision.utils.save_image(indices, 'outimg20.png')
