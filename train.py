
#import argparse
import os
#from math import log10
import torch
import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from tqdm import tqdm
from torch import nn
import torchvision.transforms as transforms

from code.dataset_cv2 import Places_dataset, save_results_img_grid4
from code.Discriminator import Discriminator, Conditional_Discriminator
from code.Generator import Generator
from code.Generator_break_63_plus_1 import GeneratorBroken63
from code.Generator_feature_extractor import GeneratorFeatures
from code.Generator_RRDB import GeneratorRRDB
from code.loss import GeneratorLoss, GeneratorLossESRGAN

import code.pytorch_ssim
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
time_stamp = datetime.now().strftime("%b%d_%H_%M")


#opt = parser.parse_args()

#CROP_SIZE = 0
UPSCALE_FACTOR = 4
NUM_EPOCHS = 21
STEPS_PER_EPOCH=10
savedata_every_epoch_num = 5
savedata_in_epoch_every_batch=100
NUM_OF_GEN_STEPS=10

lrG = 1e-4
lrD = 1e-5

out_path_sr_grid = f'./training_results/{time_stamp}/SR_results/'
out_path_params = f'./training_results/{time_stamp}/params/'
out_path_data = f'./training_results/{time_stamp}/train_data/'

if not os.path.exists(out_path_sr_grid):
    os.makedirs(out_path_sr_grid)
if not os.path.exists(out_path_params):
    os.makedirs(out_path_params)
if not os.path.exists(out_path_data):
    os.makedirs(out_path_data)
    
log_dir_name = out_path_data
tb = SummaryWriter(log_dir=log_dir_name)



def save_results_img_grid(path, gen_result, target, fname='results_grid.png', tot_images=10, nrows=2):
    red_r=gen_result[:,2:,:,:]
    blue_r=gen_result[:,0:1,:,:]
    green_r=gen_result[:,1:2,:,:]
    gen_result=torch.cat((red_r,green_r,blue_r),1)
    red_t=target[:,2:,:,:]
    blue_t=target[:,0:1,:,:]
    green_t=target[:,1:2,:,:]
    target=torch.cat((red_t,green_t,blue_t),1)
    imgsize = target.shape[2]
    datatensor = torch.empty((tot_images*2,3,imgsize,imgsize))
    res = gen_result[:tot_images,:,:,:]
    tar = target[:tot_images,:,:,:]
    datatensor[0::2,:,:,:] = tar
    datatensor[1::2,:,:,:] = res
    grid_image = utils.make_grid(datatensor, nrow=nrows, padding=3)
    utils.save_image(grid_image, os.path.join(path, fname), padding=3)

def write_to_tb(tb, epoch, results):
    tb.add_scalar('loss/generator', results['g_loss'][-1], epoch)
    tb.add_scalar('loss/discriminator', results['d_loss'][-1], epoch)
    tb.add_scalar('score/discriminator', results['g_score'][-1], epoch)
    tb.add_scalar('score/discriminator', results['d_score'][-1], epoch)
    tb.add_scalar('results/psnr', results['psnr'][-1], epoch)
    tb.add_scalar('results/ssim', results['ssim'][-1], epoch)
    
# imagenet statistics:
imagenet_mean=[0.485, 0.456, 0.406]#shay?
imagenet_std=[0.229, 0.224, 0.225]
middle_mean=[0.5, 0.5, 0.5]
middle_std=[0.5, 0.5, 0.5]

# ten_dogs_dataset_train="./data/2k_data/train"
# ten_dogs_dataset_test="./data/2k_data/test"
thirty_dogs_dataset_train="./data/6k_data/train"
thirty_dogs_dataset_test="./data/6k_data/test"

type_of_dataset="10_dogs"
spongebob_fname="frame_"
ten_dogs_fname=""

train_dataset = Places_dataset(data_path=thirty_dogs_dataset_train,indexrange=(1,5381),fname=ten_dogs_fname,type_of_dataset=type_of_dataset, transforms=transforms.Compose([transforms.Normalize(mean=middle_mean,std=middle_std)]))


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True,num_workers=4)

val_dataset = Places_dataset(data_path=thirty_dogs_dataset_test,indexrange=(1,596),fname=ten_dogs_fname,type_of_dataset=type_of_dataset,full_data=True, transforms=transforms.Compose([transforms.Normalize(mean=middle_mean,std=middle_std)]))

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False,num_workers=4)

# import models
#netG = Generator(input_nc=1, output_nc=2, num_downs=6,scale_factor=UPSCALE_FACTOR)
#netG = GeneratorBroken63(input_nc=1, output_nc=63, num_downs=6,scale_factor=UPSCALE_FACTOR)
netG=GeneratorFeatures(input_nc=1, output_nc=2, num_downs=6,scale_factor=UPSCALE_FACTOR)
#netG=GeneratorRRDB(input_nc=1, output_nc=2, num_downs=6,scale_factor=UPSCALE_FACTOR)

print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
netD = Discriminator()
#netD=Conditional_Discriminator()
print('# discriminator parameters:', sum(param.numel() for param in netD.parameters()))


#generator_criterion = GeneratorLoss()
generator_criterion = GeneratorLossESRGAN()

if torch.cuda.is_available():
    netG.cuda()
    netD.cuda()
    generator_criterion.cuda()
    print("Model on CUDA")

optimizerG = optim.Adam(netG.parameters(), lr=lrG,betas=(0.9,0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lrD,betas=(0.9,0.999))

results_dict = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}


for epoch in range(NUM_EPOCHS):
    print("folder_name",time_stamp)
    train_bar = tqdm(train_loader)
    #train_bar=tqdm(range(STEPS_PER_EPOCH))
    running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

    netG.train()
    netD.train()
    train_batch_num=-1
    for data,target in train_bar:
        #data,target=next(iter(train_loader))
        data=Variable(data)
        target=Variable(target)
        train_batch_num+=1
        if data ==None or target ==None:
            print("ERROR reading batch, skipping batch num:",train_batch_num)
            continue
        #g_update_first = True
        batch_size = data.size(0)
        running_results['batch_sizes'] += batch_size
        if torch.cuda.is_available():
            target =target.cuda()
            data = data.cuda()
        
        gray_img = data[:,0:1,:,:]
        results = netG(gray_img)
        
        ############################
        # (1) Update D network: maximize D(x)-1-D(G(z))
        ###########################
        
        netD.zero_grad()
        real_out = netD(target)
        fake_out = netD(results)
        #real_out = netD(target,gray_img)#shay?
        #fake_out = netD(results,gray_img)
        # d_loss = 1 - real_out + fake_out
        # d_loss.backward(retain_graph=True)
        # optimizerD.step()
        ### end
        
        # BCE 
        ones = torch.ones_like(real_out)
        if torch.cuda.is_available():
            ones = ones.cuda()
        real_loss = nn.BCELoss()(real_out, ones)
        fake_loss = nn.BCELoss()(fake_out, 0*ones)
        
        d_loss = real_loss + fake_loss
        d_loss.backward(retain_graph=True)
        optimizerD.step()
        

        ############################
        # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
        ###########################
        netG.zero_grad()
        # WRONG? :
        #g_loss = generator_criterion(fake_out, results, target)
        #g_loss.backward()
        #fake_img = netG(data)
        #fake_out = netD(fake_img).mean()
        
        # correct?
        fake_out = netD(results)
        #fake_out = netD(results,gray_img)
        #fool_GAN_loss = nn.BCELoss()(fake_out, ones)  # fool the discriminator
        perceptual_loss  = generator_criterion(fake_out, results, target)
        
        #g_loss = perceptual_loss + fool_GAN_loss
        g_loss = perceptual_loss
        g_loss.backward()
        optimizerG.step()

        # loss for current batch before optimization 
        running_results['g_loss'] += g_loss.item() * batch_size
        running_results['d_loss'] += d_loss.item() * batch_size
        running_results['d_score'] += real_out.mean().item() * batch_size
        running_results['g_score'] += fake_out.mean().item() * batch_size
       
        #print("the problem:",type(running_results['d_loss']) ,  type(running_results['batch_sizes']))
       

        losd = running_results['d_loss'] / running_results['batch_sizes']
        losg = running_results['g_loss'] / running_results['batch_sizes']
        scrd = running_results['d_score'] / running_results['batch_sizes']
        scrg = running_results['g_score'] / running_results['batch_sizes']
        desc = f'[{epoch}/{NUM_EPOCHS}], Loss_D: {losd:.3f} Loss_G: {losg:.3f} D(x): {scrd:.3f} D(G(z)): {scrg:.3f}'
        train_bar.set_description(desc=desc)

        if train_batch_num% savedata_in_epoch_every_batch==0:
            save_results_img_grid(out_path_sr_grid,results,target,fname=f'results_{epoch}_iter{train_batch_num}.png')
        
          
        
    
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
            results = netG(val_data[:,0:1,:,:])
    
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
                save_results_img_grid4(out_path_sr_grid,results,val_target,bicub_imgs,input_imgs,fname=f'results_val_{epoch}_iter{train_batch_num}.png')

        #shay? psnr computation:
        val_results['psnr'] = 10 * torch.log10((val_target.max()**2) / (val_results['mse'] / val_results['batch_sizes']))
        val_results['ssim'] = val_results['ssims'] / val_results['batch_sizes']    
    
        #val_bar.set_description(desc=f"[Validation] PSNR:{val_results['psnr']:.4f} dB SSIM: {val_results['ssim']:.4f}")
    
    
    # save loss\scores\psnr\ssim
    # continue
    results_dict['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
    results_dict['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
    results_dict['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
    results_dict['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
    results_dict['psnr'].append(val_results['psnr'])
    results_dict['ssim'].append(val_results['ssim'])
    
    write_to_tb(tb, epoch, results_dict)

    if epoch % savedata_every_epoch_num == 0 :#and epoch != 0:
        #save_results_img_grid(out_path_sr_grid, 
        #                      torch.cat([results, save_prev_res], 0),
        #                      torch.cat([val_target, save_prev_tar], 0),
        #                      fname=f'results_{epoch}.png')
        # save model parameters
        torch.save(netG.state_dict(), os.path.join(out_path_params, f'netG_epoch_{epoch}.pth'))
        torch.save(netD.state_dict(), os.path.join(out_path_params, f'netD_epoch_{epoch}.pth'))
        
        data_frame = pd.DataFrame(data={'Loss_D': results_dict['d_loss'], 'Loss_G': results_dict['g_loss'], 'Score_D': results_dict['d_score'],
                  'Score_G': results_dict['g_score'], 'PSNR': results_dict['psnr'], 'SSIM': results_dict['ssim']},index=range(epoch+1))
        data_frame.to_csv( os.path.join(out_path_data, 'SR_train_results.csv'), index_label='epoch')
        


tb.close()







