import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import glob
from dataloader import Rescale
from dataloader import RescaleT
from dataloader import RandomCrop
from dataloader import ToTensor
from dataloader import ToTensorLab
from dataloader import LoadDataset
from model import ures2net
# Credits: https://github.com/xuebinqin/U-2-Net
# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)

    return dn

def save_output(image_name,pred,d_dir):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)
    pb_np = np.array(imo)

    name_data = img_name.split(".")
    idx = name_data[0:-1]
    imidx = idx[0]
    for i in range(1,len(idx)):
        imidx = imidx + "." + idx[i]

    imo.save(d_dir+imidx+'.png')

# --------- 1. get image path and name ---------

model_name='ures2net'
image_dir = os.path.join('ISIC_2017/Test/ISIC-2017_Test_v2_Data/')
prediction_dir = os.path.join('ISIC_2017/test_results_ures2net/', model_name + '_results_on_ISIC2017_epoch100' + os.sep)
model_dir = "ISIC_2017/my_model_weights/ures2net_ISIC2017_epoch_100_train_0.371395_tar_0.050255.pth"
img_name_list = glob.glob(image_dir + os.sep + '*')
print(img_name_list)
# --------- 2. dataloader ---------
#1. dataloader
test_dataset = LoadDataset(img_name_list = img_name_list,
                           lbl_name_list = [],
                           transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)]))
test_dataloader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=1)
# --------- 3. model define ---------
def test():
    if(model_name=='ures2net'):
      net = ures2net(3,1)

    if torch.cuda.is_available():
      print("cuda is available.")
      net.load_state_dict(torch.load(model_dir))
      net.cuda()
    else:
      checkpoint_older = torch.load(model_dir, map_location='cpu')
      net.load_state_dict(checkpoint_older['model'])

    net.eval()
    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_dataloader):
         print("Inferencing:",img_name_list[i_test].split(os.sep)[-1])
         inputs_test = data_test['image']
         inputs_test = inputs_test.type(torch.FloatTensor)
         if torch.cuda.is_available():
             inputs_test = Variable(inputs_test.cuda())
         else:
             inputs_test = Variable(inputs_test)
         d1,d2,d3,d4,d5,d6,d7= net(inputs_test)
         # normalization
         pred = d1[:,0,:,:]
         pred = normPRED(pred)
         # save results to test_results folder
         if not os.path.exists(prediction_dir):
             os.makedirs(prediction_dir, exist_ok=True)
         save_output(img_name_list[i_test],pred,prediction_dir)
         del d1,d2,d3,d4,d5,d6,d7

if __name__ == '__main__':
    test()
