
        
#%% Dataset:
    
import torch
#import torch.nn as nn
#import torch.optim as optim
#import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
#import torchvision.transforms as transforms
from torchvision import transforms
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
import torchvision.utils as utils

class Places_dataset(Dataset):
    
    def __init__(self, data_path = '', fname='Places365_val_', type_of_dataset="places365",
                 indexrange=(1,36500),transforms=None, full_data=False):
        
        self.data_path = data_path
        self.fname = fname
        self.transforms = transforms
        self.indexrange = indexrange
        self.data_size = indexrange[1]-indexrange[0]+1 # include edges!
        self.type_of_dataset=type_of_dataset
        if self.data_size <= 0:
            raise ValueError('fix the dataset range')
        self.full_data_ind = full_data
    
    def __len__(self):
        return self.data_size
      
    def __getitem__(self, index):
        img_index = index+self.indexrange[0]
        if self.type_of_dataset=="spongebob":
            file_name=self.fname+f'{img_index:0=4d}'+'.png'
        elif self.type_of_dataset=="10_dogs":
            file_name=self.fname+f'{img_index}.jpg'
        else:
            file_name = self.fname + f'{img_index:0=8d}' + '.jpg'
        
        #im = Image.open(self.data_path + file_name)
        img = cv2.imread(os.path.join(self.data_path,file_name))
        if not isinstance(img,np.ndarray):
            print(file_name)
            file_name=self.fname+"0001.png"
            img=cv2.imread(os.path.join(self.data_path,file_name))
        if img.shape[0] !=256 or img.shape[1] !=256:
            img = cv2.resize(img, (256,256))
        img_tensor = np.transpose(img/255, (2,0,1))
        img_tensor = torch.from_numpy(img_tensor).float()
        
        small = cv2.resize(img, (64,64))
        lab_small = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lab2_small = np.transpose(lab_small/255, (2,0,1))
        lab3_small = torch.from_numpy(lab2_small).float()
        lab3_small=self.transforms(lab3_small)

        #gray_im = im.convert('L')
        
        if self.full_data_ind:
            bicubic_inter = cv2.resize(small, (256,256), interpolation=cv2.INTER_CUBIC)
            bicubic_inter = torch.from_numpy(np.transpose(bicubic_inter/255, (2,0,1))).float()
            input_emulation = cv2.cvtColor(lab_small[:,:,0],cv2.COLOR_GRAY2RGB)
            input_emulation = cv2.resize(input_emulation, (256,256), interpolation=cv2.INTER_NEAREST)
            input_emulation = torch.from_numpy(np.transpose(input_emulation/255, (2,0,1))).float()
            return (lab3_small, img_tensor, bicubic_inter, input_emulation)
        
       # if self.transforms is not None:
            #lab3_small = self.transforms(lab3_small)
       #     img_tensor=self.transforms(img_tensor)
            
       # else: 
       #     img_tensor=img_tensor-torch.ones_like(img_tensor)

        return (lab3_small, img_tensor)




train_dataset = Places_dataset(data_path = 'val_256', indexrange=(5,20), transforms=transforms.Compose([
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5]) ]), full_data=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

def bgr_to_rgb(pytensor):
    red=pytensor[:,2:,:,:]
    green=pytensor[:,1:2,:,:]
    blue=pytensor[:,0:1,:,:]
    return torch.cat((red,green,blue),1)

def save_results_img_grid4(path, gen_result, target, bicub, input_imgs, fname='results_grid.png', tot_images=10, nrows=4):
    gen_result=bgr_to_rgb(gen_result)
    target=bgr_to_rgb(target)
    bicub=bgr_to_rgb(bicub)
    input_imgs=bgr_to_rgb(input_imgs)
    imgsize = target.shape[2]
    datatensor = torch.empty((tot_images*4,3,imgsize,imgsize))
    res = gen_result[:tot_images,:,:,:]
    tar = target[:tot_images,:,:,:]
    bicub = bicub[:tot_images,:,:,:]
    inpt = input_imgs[:tot_images,:,:,:]
    datatensor[0::4,:,:,:] = inpt
    datatensor[1::4,:,:,:] = bicub
    datatensor[2::4,:,:,:] = res
    datatensor[3::4,:,:,:] = tar
    grid_image = utils.make_grid(datatensor, nrow=nrows, padding=3)
    utils.save_image(grid_image, os.path.join(path, fname), padding=3)

