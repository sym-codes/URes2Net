import os
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob

from dataloader import Rescale
from dataloader import RescaleT
from dataloader import RandomCrop
from dataloader import ToTensor
from dataloader import ToTensorLab
from dataloader import LoadDataset
from model import ures2net
# Credits: https://github.com/xuebinqin/U-2-Net
# ------- 1. define loss function --------
bce_loss = nn.BCELoss(size_average=True)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)
    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    return loss0, loss

# ------- 2. set the directory of training dataset --------
model_name = 'ures2net'
tra_image_dir = os.path.join('ISIC_2017/Train/ISIC-2017_Training_Data/')
tra_label_dir = os.path.join('ISIC_2017/Train/ISIC-2017_Training_Part1_GroundTruth/')

image_ext = '.jpg'
label_ext = '.png'

model_dir = os.path.join('ISIC_2017/my_model_weights/')

tra_img_name_list = glob.glob(tra_image_dir + '*' + image_ext)
tra_lbl_name_list = glob.glob(tra_label_dir + '*' + label_ext)


def train(epoch_num=1000, batch_size_train=5, batch_size_val=1, train_num=0, val_num=0):
	train_num = len(tra_img_name_list)
	seg_dataset = LoadDataset(
		img_name_list=tra_img_name_list,
		lbl_name_list=tra_lbl_name_list,
		transform=transforms.Compose([
			RescaleT(320),
			RandomCrop(288),
			ToTensorLab(flag=0)]))

	seg_dataloader = DataLoader(seg_dataset, batch_size=batch_size_train, shuffle=True, num_workers=1)

	# ------- 4. define optimizer --------
	print("---define optimizer...")
	optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

	# ------- 5. training process --------
	print("---start training...")
	ite_num = 0
	running_loss = 0.0
	running_tar_loss = 0.0
	ite_num4val = 0
	save_frq = 123

	# To load old trained parameters and continue training:
    # enter your path inormation and set continue_training variable as True
	old_pth = ""
	continue_training = False
	if continue_training:
		print("Old weight is loaded.")
		checkpoint_older = torch.load(old_pth)
		net.load_state_dict(checkpoint_older['model'])
		optimizer.load_state_dict(checkpoint_older['optimizer'])
		epoch_old = checkpoint_older['epoch']
		loss = checkpoint_older['loss']
	else:
		epoch_old = 0

	for epoch in range(0, epoch_num):

		net.train()

		for i, data in enumerate(seg_dataloader):
			ite_num = ite_num + 1
			ite_num4val = ite_num4val + 1

			inputs, labels = data['image'], data['label']

			inputs = inputs.type(torch.FloatTensor)
			labels = labels.type(torch.FloatTensor)

			# wrap them in Variable
			if torch.cuda.is_available():
				inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
																							requires_grad=False)
			else:
				inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

			# y zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
			loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.data.item()
			running_tar_loss += loss2.data.item()

			# del temporary outputs and loss
			del d0, d1, d2, d3, d4, d5, d6, loss2, loss

		PATH = model_dir + "ures2net_ISIC2017_epoch_%d_train_%3f_tar_%3f.pth" % (
		epoch + 1 + epoch_old, running_loss / ite_num4val, running_tar_loss / ite_num4val)
		checkpoint = {
			'epoch': (epoch + 1 + epoch_old),
			'model': net.state_dict(),
			'optimizer': optimizer.state_dict(),
			'loss': (running_loss / ite_num4val)}
		torch.save(checkpoint, PATH)

		print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
			epoch + 1 + epoch_old, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val,
			running_tar_loss / ite_num4val))

		running_loss = 0.0
		running_tar_loss = 0.0
		net.train()
		ite_num4val = 0

if __name__ == '__main__':
	# ------- 3. define model --------
	# define the model
	if (model_name == 'ures2net'):
		net = ures2net(3, 1)

	if torch.cuda.is_available():
		net.cuda()

	train(epoch_num = 700, batch_size_train = 5, batch_size_val = 1, train_num = 0, val_num = 0)

