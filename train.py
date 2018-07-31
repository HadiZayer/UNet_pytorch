import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from torch.autograd import Variable
from torch import nn
from augmentation import augment_data
import numpy as np
import torchvision
from loss import MulticlassDiceLoss


def oneHotEncoding(labels, num_classes):
    N, H, W = labels.size()
    output = torch.zeros((N, num_classes, H, W))
    output = output.scatter_(1, labels.unsqueeze(1).long(), 1)
    return output

def train(model, images, labels, batch_size, epochs, num_classes, lr=0.1, gpu=True):
    
    oneHotLabels = oneHotEncoding(labels, num_classes)

    if(gpu):
        model.cuda()
        images = images.cuda()
        oneHotLabels = oneHotLabels.cuda()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3)
    criterion = MulticlassDiceLoss()
    mse = nn.MSELoss()


    dataset = torch.utils.data.TensorDataset(images, oneHotLabels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    iters = 0


    for epoch in range(epochs):
        epoch_loss = 0
        for batch, target in dataloader:

            if(len(target[:,1:].nonzero()) == 0):
                continue


            batch_var = Variable(batch)
            
            target_var = Variable(target)

    
            prediction = model(batch_var)
            softmax = nn.Softmax2d()
            soft_prediction = softmax(prediction)

            diceLoss = criterion(soft_prediction, target_var, ignore_indices=[0])


            loss = diceLoss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            epoch_loss += loss

        
        print('loss: {}'.format(epoch_loss))