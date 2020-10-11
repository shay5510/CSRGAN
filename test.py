
#import argparse
import os
#from math import log10
import torch
import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
#from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn

import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import math
#import pytorch_ssim
#from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
#from loss import GeneratorLoss
#from model import Generator, Discriminator

import torchvision.transforms as transforms
from code.dataset_cv2 import Places_dataset, save_results_img_grid4
from code.Discriminator import Discriminator, Conditional_Discriminator
from code.Generator_feature_extractor import GeneratorFeatures,UpsampleBLock,UnetSkipConnectionBlock
from code.loss import GeneratorLoss, GeneratorLossESRGAN,bgr_to_rgb_imagenet_normalized

import code.pytorch_ssim
from datetime import datetime
#from torch.utils.tensorboard import SummaryWriter


class with_center(GeneratorFeatures):

    def forward(self, input):
        # x=self.model(input)
        x=torch.cat([input, self.model(input)], 1)
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)
        return ((torch.tanh(block8) + 1) / 2), x
#opt = parser.parse_args()

#CROP_SIZE = 0
UPSCALE_FACTOR = 4
NUM_EPOCHS = 21
STEPS_PER_EPOCH=10
savedata_every_epoch_num = 5
savedata_in_epoch_every_batch=100
NUM_OF_GEN_STEPS=10


erezout_path_detailed = f'./test_results/'


if not os.path.exists(erezout_path_detailed):
    os.makedirs(erezout_path_detailed)
    
def ttg(tensor):
    pil = transforms.ToPILImage()(tensor)
    pil = pil.convert('L')
    return transforms.ToTensor()(pil)

def x_process(x):
    pil = transforms.ToPILImage()((x+1)/2)
    n = np.asarray(pil)
    nrgb = cv2.cvtColor(n, cv2.COLOR_LAB2RGB)
    check = cv2.cvtColor(n[:,:,0], cv2.COLOR_GRAY2RGB)
    
    from PIL import Image
    pilrgb = Image.fromarray(nrgb)
    pilc = Image.fromarray(check)
    return transforms.ToTensor()(pil),transforms.ToTensor()(pilc)
#log_dir_name = out_path_data
def bgr_to_rgb_img(pytensor):
    red=pytensor[:,2:,:,:]
    green=pytensor[:,1:2,:,:]
    blue=pytensor[:,0:1,:,:]
    return torch.cat((red,green,blue),1)
#save_results_img_grid4(path, gen_result, target, bicub, input_imgs, fname='results_grid.png', tot_images=10, nrows=4):
def save_result_for_img(path, gen_result, target, bicub, input_imgs, x, idata, i):
    gen_result=bgr_to_rgb_img(gen_result)
    target=bgr_to_rgb_img(target)
    bicub=bgr_to_rgb_img(bicub)
    input_imgs=bgr_to_rgb_img(input_imgs)
    imgsize = target.shape[2]
    
    xrgb,check = x_process(x)
    
    datatensor = torch.empty((4,3,imgsize,imgsize))
    
    datatensor[0,:,:,:] = input_imgs
    datatensor[1,:,:,:] = bicub
    datatensor[2,:,:,:] = gen_result
    datatensor[3,:,:,:] = target
    grid_image = utils.make_grid(datatensor, nrow=4, padding=20,pad_value=1)
    utils.save_image(grid_image, os.path.join(path, f'{i} grid.png'), padding=3)
    utils.save_image(target, os.path.join(path, f'{i} target.png'), padding=3)
    utils.save_image(bicub, os.path.join(path, f'{i} bicubic.png'), padding=3)
    utils.save_image(gen_result, os.path.join(path, f'{i} result.png'), padding=3)
    utils.save_image(input_imgs, os.path.join(path, f'{i} input.png'), padding=3)
    #utils.save_image(xrgb, os.path.join(path, f'{i} x_rgb.png'), padding=3)
    if i<=1:
        utils.save_image(check, os.path.join(path, f'{i} check.png'), padding=3)
    mse1 = ((gen_result - target) ** 2).mean()
    mse2 = ((bicub - target) ** 2).mean()
    idata['i'].append(i)
    idata['mse_result'].append(mse1)
    idata['mse_bicub'].append(mse2)
    idata['psnr_result'].append( 10 * torch.log10((target.max()**2) / mse1))
    idata['psnr_bicubic'].append( 10 * torch.log10((target.max()**2) / mse2))
    mse11 = ((ttg(gen_result) - ttg(target)) ** 2).mean()
    mse22 = ((ttg(bicub) - ttg(target)) ** 2).mean()
    idata['mse_result_gray'].append(mse11)
    idata['mse_bicub_gray'].append(mse22)
    idata['psnr_result_gray'].append( 10 * torch.log10((ttg(target).max()**2) / mse11))
    idata['psnr_bicubic_gray'].append( 10 * torch.log10((ttg(target).max()**2) / mse22))
    crit = VGG_VALUE()
    idata['vgg_result'].append(crit(torch.unsqueeze(gen_result, 0),torch.unsqueeze(target, 0)))
    idata['vgg_bicubic'].append(crit(torch.unsqueeze(bicub, 0),torch.unsqueeze(target, 0)))
    

    
# imagenet statistics:
imagenet_mean=[0.485, 0.456, 0.406]#shay?
imagenet_std=[0.229, 0.224, 0.225]
middle_mean=[0.5, 0.5, 0.5]
middle_std=[0.5, 0.5, 0.5]


#
# ten_dogs_dataset_train="./data/2k_data/train"
# ten_dogs_dataset_test="./data/2k_data/test"
thirty_dogs_dataset_train="./data/6k_data/train"
thirty_dogs_dataset_test="./data/6k_data/test"

type_of_dataset="10_dogs"
spongebob_fname="frame_"
ten_dogs_fname=""

#train_dataset = Places_dataset(data_path=thirty_dogs_dataset_train,indexrange=(1,5381),fname=ten_dogs_fname,type_of_dataset=type_of_dataset, transforms=transforms.Compose([transforms.Normalize(mean=middle_mean,std=middle_std)]))


#train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True,num_workers=4)
val_dataset = Places_dataset(data_path=thirty_dogs_dataset_test,indexrange=(1,596),fname=ten_dogs_fname,type_of_dataset=type_of_dataset,full_data=True, transforms=transforms.Compose([transforms.Normalize(mean=middle_mean,std=middle_std)]))

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False,num_workers=4)


netG=with_center(input_nc=1, output_nc=2, num_downs=6,scale_factor=UPSCALE_FACTOR)
params_path = 'params/6k_params/netG_epoch_20.pth'
netG.load_state_dict(torch.load(params_path))
#netG=GeneratorRRDB(input_nc=1, output_nc=2, num_downs=6,scale_factor=UPSCALE_FACTOR)

print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
#netD = Discriminator()
#netD=Conditional_Discriminator()
#print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))

class VGG_VALUE(GeneratorLossESRGAN):
    def forward(self, out_images, target_images):
        
        #imagenet preparation
        out_images_VGG=bgr_to_rgb_imagenet_normalized(out_images)
        target_images_VGG=bgr_to_rgb_imagenet_normalized(target_images)

        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images_VGG), self.loss_network(target_images_VGG))
        return perception_loss



#generator_criterion = GeneratorLoss()
generator_criterion = GeneratorLossESRGAN()

if torch.cuda.is_available():
    netG.cuda()
    #netD.cuda()
    generator_criterion.cuda()
    print("Model on CUDA")

#optimizerG = optim.Adam(netG.parameters(), lr=lrG,betas=(0.9,0.999))
#optimizerD = optim.Adam(netD.parameters(), lr=lrD,betas=(0.9,0.999))

results_dict = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}


print("folder_name",time_stamp)
idata = {}
## Validation:
netG.eval()

with torch.no_grad():
    val_bar = tqdm(val_loader)
    val_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
    val_images = []
    val_batch_num=-1
    for val_data, val_target,bicub_imgs,input_imgs in val_bar:
        val_batch_num += 1
        if val_data == None or val_target == None:
            print("ERROR reading batch, skipping batch num:", val_batch_num)
            continue
        batch_size = val_data.size(0)
        val_results['batch_sizes'] += batch_size
        
        if torch.cuda.is_available():
            val_data = val_data.cuda()
            val_target = val_target.cuda()
        results,x = netG(val_data[:,0:1,:,:])

        batch_mse = ((results - val_target) ** 2).mean()
        val_results['mse'] += batch_mse * batch_size
        
        batch_ssim = pytorch_ssim.ssim(results, val_target).item() #shay?ssim computation..
        val_results['ssims'] += batch_ssim * batch_size
        
        
        
        tmp_psnr=10 * torch.log10((val_target.max()**2) / (val_results['mse'] / val_results['batch_sizes']))
        tmp_ssim= val_results['ssims'] / val_results['batch_sizes']    
        
        #val_bar.set_description(desc=f"[Validation] PSNR:{val_results['psnr']:.4f} dB SSIM: {val_results['ssim']:.4f}")
        val_bar.set_description(desc=f"[Validation] PSNR:{tmp_psnr:.4f} dB SSIM: {tmp_ssim:.4f}")
       
        save_prev_res = results
        save_prev_tar = val_target
          
        if val_batch_num==1:
            
            idata['i'] = []
            idata['mse_result'] = []
            idata['mse_bicub'] = []
            idata['psnr_result'] = []
            idata['psnr_bicubic'] = []
            idata['mse_result_gray'] = []
            idata['mse_bicub_gray'] = []
            idata['psnr_result_gray'] = []
            idata['psnr_bicubic_gray'] = []
            idata['vgg_result'] = []
            idata['vgg_bicubic'] = []
            for i in range(batch_size):
                save_result_for_img(erezout_path_detailed, results[i], val_target[i], bicub_imgs[i], input_imgs[i], x[i], idata, i )
            break

    #shay? psnr computation:
    #val_results['psnr'] = 10 * torch.log10((val_target.max()**2) / (val_results['mse'] / val_results['batch_sizes']))
    #val_results['ssim'] = val_results['ssims'] / val_results['batch_sizes']    

    #val_bar.set_description(desc=f"[Validation] PSNR:{val_results['psnr']:.4f} dB SSIM: {val_results['ssim']:.4f}")

data_frame = pd.DataFrame(data=idata)
data_frame.to_csv( os.path.join(out_path_data, 'validation data.csv'))
# save loss\scores\psnr\ssim
# continue

results_dict['psnr'].append(val_results['psnr'])
results_dict['ssim'].append(val_results['ssim'])

        









