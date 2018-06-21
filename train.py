import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from torch.autograd import Variable
from torch import nn
import loss
from augmentation import augment_data

def oneHotEncoding(labels, num_classes):
    N, H, W = labels.size()
    output = torch.zeros((N, num_classes, H, W))
    output = output.scatter_(1, labels.unsqueeze(1).long(), 1)
    return output

def train(model, images, labels, batch_size, epochs, lr=0.1, gpu=False):
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = loss.MulticlassDiceLoss()
    
    if(gpu):
        model.cuda()
        images = images.cuda()
        labels = labels.cuda()

    dataset = torch.utils.data.TensorDataset(images, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    iters = 0

    for epoch in range(epochs):
        epoch_loss = 0
        for batch, target in dataloader:
            for b,t in zip(batch, target):
                t3 = torch.stack([t,t,t])
                augmented_b, augmented_t = augment_data(b, t3, rotationNum=1, cropNum=1, hflip=True, vflip=True)
                augmented_t = augmented_t[:,0,:,:]

                augmented_t = oneHotEncoding(augmented_t, 2)

                batch_var = Variable(augmented_b)
                target_var = Variable(augmented_t)



        # x = Variable(torch.FloatTensor(np.random.random((2, 1, 256, 256))))
            
        
                prediction = model(batch_var)
                sigmoid = nn.Sigmoid()
                prediction = sigmoid(prediction)
                diceloss = criterion(prediction, target_var)
                # bce = nn.BCELoss()
                # loss += bce(prediction, target_var)
                optimizer.zero_grad()
                diceloss.backward()
                optimizer.step()
            
                epoch_loss += diceloss
                print(diceloss)

            # iters += 1
            # if(iters == 2):
            #     break
        # gc.collect()
        # del x, pred_masks
        
        print('Epoch {}, loss: {}'.format(epoch, epoch_loss))