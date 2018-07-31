import os
from skimage import io
from model.unet import unet, small_unet
import torchvision
import torch
import train
import numpy as np
import brats_loader
from torch.autograd import Variable


path = os.path.dirname(os.getcwd())
path = os.path.join(path, 'BRATS_data\\test')

t1, t1gd, t2, flair, labels = brats_loader.load_data(path)

for scans in (t1, t1gd, t2, flair):
	for scan in scans:
		brats_loader.preprocess(scan)


test_scans = np.stack([t1, t1gd, t2, flair])
# original scans shape = CxNxHxWxD
# want NxDxCxHxW
test_scans = test_scans.transpose(1, 4, 0, 2, 3)




# original labels shape = NxHxWxD
# want NxDxHxW
test_labels = np.array(labels)
test_labels[test_labels!=0] = 1
test_labels = test_labels.transpose(0 , 3, 1, 2)

test_scans = torch.Tensor(test_scans)
test_labels = torch.Tensor(test_labels).long()

#  NxDxCxHxW, NxDxCxHxW
scans_config = [(1, 4, 0, 2, 3)]#, (1, 4, 0, 3, 2)]
#xDxHxW, NxDxHxW
labels_config = [ (0 , 3, 1, 2)]#, (0 , 3, 2, 1)]

model = small_unet(4,2, dropout=True)
# model.load_state_dict(torch.load('small_unet_outputs\\brats_model_dice_v01.pt'))

for j in range(100):

	ids = np.arange(10)
	np.random.shuffle(ids)
	for i, k in enumerate(ids):

		for sc, lc in zip(scans_config, labels_config):
			path = os.path.dirname(os.getcwd())
			path = os.path.join(path, 'BRATS_data\\sample' + str(k))

			t1, t1gd, t2, flair, labels = brats_loader.load_data(path)

			for scans in (t1, t1gd, t2, flair):
				for scan in scans:
					brats_loader.preprocess(scan)


			scans = np.stack([t1, t1gd, t2, flair])
			# original scans shape = CxNxHxWxD
			# want NxDxCxHxW
			scans = scans.transpose(sc[0], sc[1], sc[2], sc[3], sc[4])




			# original labels shape = NxHxWxD
			# want NxDxHxW
			labels = np.array(labels)
			labels[labels!=0] = 1
			labels = labels.transpose(lc[0] , lc[1], lc[2], lc[3])

			scans = torch.Tensor(scans)
			labels = torch.Tensor(labels).long()

			
			# model.load_state_dict(torch.load('brats_model.pt'))


			# for scan, label in zip(scans, labels):
			scans = [scan for scan in scans]
			labels = [label for label in labels]

			scans = torch.cat(scans, dim=0)
			labels = torch.cat(labels, dim=0)

			train.train(model, scans, labels, batch_size=1, epochs=1, lr=1e-2, num_classes=2, gpu=True)
	torch.save(model.state_dict(), 'small_unet_outputs\\brats_model_dice_v0' + str(j) + '.pt')

	testimg = test_scans[0,60]
	testimg = testimg.unsqueeze(0)
	testimg = Variable(testimg.cuda())
	out = model(testimg)
	_, indices = torch.max(out, 1)
	torchvision.utils.save_image(test_scans[0,60,0], 'small_unet_outputs\\test' + str(i) + str(j) + '0.png')
	torchvision.utils.save_image(test_labels[0,60], 'small_unet_outputs\\label' + str(i) + str(j) + '0.png')
	torchvision.utils.save_image(indices, 'small_unet_outputs\\out_test' + str(i) + str(j) + '0.png')

	testimg = test_scans[1,60]
	testimg = testimg.unsqueeze(0)
	testimg = Variable(testimg.cuda())
	out = model(testimg)
	_, indices = torch.max(out, 1)
	torchvision.utils.save_image(test_scans[1,60,0], 'small_unet_outputs\\test'+ str(i) + str(j) + '1.png')
	torchvision.utils.save_image(test_labels[1,60], 'small_unet_outputs\\label'+ str(i) + str(j) + '1.png')
	torchvision.utils.save_image(indices, 'small_unet_outputs\\out_test'+ str(i) + str(j) + '1.png')

	testimg = test_scans[2,60]
	testimg = testimg.unsqueeze(0)
	testimg = Variable(testimg.cuda())
	out = model(testimg)
	_, indices = torch.max(out, 1)
	torchvision.utils.save_image(test_scans[2,60,0], 'small_unet_outputs\\test'+ str(i) + str(j) + '2.png')
	torchvision.utils.save_image(test_labels[2,60], 'small_unet_outputs\\label'+ str(i) + str(j) + '2.png')
	torchvision.utils.save_image(indices, 'small_unet_outputs\\out_test'+ str(i) + str(j) + '2.png')

