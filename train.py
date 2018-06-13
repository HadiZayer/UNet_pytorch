import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from torch.autograd import Variable
from torch import nn

def train(model, images, labels, batch_size, epochs, lr=0.1, gpu=False):
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    if(gpu):
        model.cuda()
        images = images.cuda()
        labels = labels.cuda()

    dataset = torch.utils.data.TensorDataset(images, labels)
    dataloader = torch.utils.data.DataLoader(dataset)
    
    iters = 0

    for epoch in range(epochs):
        epoch_loss = 0
        for batch, target in dataloader:
            batch_var = Variable(batch)
            target_var = Variable(target)



        # x = Variable(torch.FloatTensor(np.random.random((2, 1, 256, 256))))
            
        
            prediction = model(batch_var)
            loss = criterion(prediction, target_var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss
            print(loss)
            # iters += 1
            # if(iters == 5):
            #     break
        # gc.collect()
        # del x, pred_masks
        
        print('Epoch {}, loss: {}'.format(epoch, epoch_loss))