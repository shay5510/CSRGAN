# Authors Erez Yosef & Shay Shomer Chai
# all rights reserved

import os
import torch
import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from tqdm import tqdm
from torch import nn
import numpy as np
import cv2
import argparse

import torchvision.transforms as transforms
import sys
sys.path.append("./code")
from dataset_cv2 import Places_dataset, save_results_img_grid4
from Discriminator import Discriminator, Conditional_Discriminator
from Generator_feature_extractor import GeneratorFeatures,UpsampleBLock,UnetSkipConnectionBlock
from loss import GeneratorLoss, GeneratorLossESRGAN,bgr_to_rgb_imagenet_normalized

import pytorch_ssim
from constants import *


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


class VGG_VALUE(GeneratorLossESRGAN):
    def forward(self, out_images, target_images):
        
        #imagenet preparation
        out_images_VGG=bgr_to_rgb_imagenet_normalized(out_images)
        target_images_VGG=bgr_to_rgb_imagenet_normalized(target_images)

        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images_VGG), self.loss_network(target_images_VGG))
        return perception_loss

class CSRGANTest():
    def __init__(self,args):
        self.train_path=args.train_path
        self.test_path=args.test_path
        self.batch_size=args.batch_size
        self.type_of_dataset=args.type_of_dataset
        self.fname=args.fname
        self.params_path = args.params_path
        self.init_paths()
        self.prepare_dataset()


    def init_paths(self):
        self.test_dir= './test_results/'

        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

    def prepare_dataset(self):
        val_dataset = Places_dataset(data_path=self.test_path, indexrange=(1, 595), fname=self.fname,
                                     type_of_dataset=self.type_of_dataset, full_data=True,
                                     transforms=transforms.Compose(
                                         [transforms.Normalize(mean=middle_mean, std=middle_std)]))

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        self.val_loader=val_loader

    def ttg(self,tensor):
        pil = transforms.ToPILImage()(tensor.cpu())
        pil = pil.convert('L')
        return transforms.ToTensor()(pil)

    def x_process(self,x):
        x = x.cpu()
        pil = transforms.ToPILImage()((x + 1) / 2)
        n = np.asarray(pil)
        nrgb = cv2.cvtColor(n, cv2.COLOR_LAB2RGB)
        check = cv2.cvtColor(n[:, :, 0], cv2.COLOR_GRAY2RGB)

        from PIL import Image
        pilrgb = Image.fromarray(nrgb)
        pilc = Image.fromarray(check)
        return transforms.ToTensor()(pil), transforms.ToTensor()(pilc)

    # log_dir_name = out_path_data
    def bgr_to_rgb_img(self,pytensor):
        red = pytensor[2:, :, :]
        green = pytensor[1:2, :, :]
        blue = pytensor[0:1, :, :]
        return torch.cat((red, green, blue), 0)

    # save_results_img_grid4(path, gen_result, target, bicub, input_imgs, fname='results_grid.png', tot_images=10, nrows=4):
    def save_result_for_img(path, gen_result, target, bicub, input_imgs, x, idata, i):
        gen_result = self.bgr_to_rgb_img(gen_result)
        target = self.bgr_to_rgb_img(target)
        bicub = self.bgr_to_rgb_img(bicub)
        input_imgs = self.bgr_to_rgb_img(input_imgs)
        imgsize = target.shape[2]
        bicub = bicub.cuda()
        xrgb, check = self.x_process(x)

        datatensor = torch.empty((4, 3, imgsize, imgsize))

        datatensor[0, :, :, :] = input_imgs
        datatensor[1, :, :, :] = bicub
        datatensor[2, :, :, :] = gen_result
        datatensor[3, :, :, :] = target
        grid_image = utils.make_grid(datatensor, nrow=4, padding=20, pad_value=1)
        utils.save_image(grid_image, os.path.join(path, f'{i} grid.png'), padding=3)
        utils.save_image(target, os.path.join(path, f'{i} target.png'), padding=3)
        utils.save_image(bicub, os.path.join(path, f'{i} bicubic.png'), padding=3)
        utils.save_image(gen_result, os.path.join(path, f'{i} result.png'), padding=3)
        utils.save_image(input_imgs, os.path.join(path, f'{i} input.png'), padding=3)
        # utils.save_image(xrgb, os.path.join(path, f'{i} x_rgb.png'), padding=3)
        if i <= 1:
            utils.save_image(check, os.path.join(path, f'{i} check.png'), padding=3)
        mse1 = ((gen_result - target) ** 2).mean()
        mse2 = ((bicub - target) ** 2).mean()
        idata['i'].append(i)
        idata['mse_result'].append(mse1)
        idata['mse_bicub'].append(mse2)
        idata['psnr_result'].append(10 * torch.log10((target.max() ** 2) / mse1))
        idata['psnr_bicubic'].append(10 * torch.log10((target.max() ** 2) / mse2))
        mse11 = ((self.ttg(gen_result) - self.ttg(target)) ** 2).mean()
        mse22 = ((self.ttg(bicub) - self.ttg(target)) ** 2).mean()
        idata['mse_result_gray'].append(mse11)
        idata['mse_bicub_gray'].append(mse22)
        idata['psnr_result_gray'].append(10 * torch.log10((self.ttg(target).max() ** 2) / mse11))
        idata['psnr_bicubic_gray'].append(10 * torch.log10((self.ttg(target).max() ** 2) / mse22))
        crit = VGG_VALUE().cuda()
        idata['vgg_result'].append(crit(torch.unsqueeze(gen_result, 0), torch.unsqueeze(target, 0)))
        idata['vgg_bicubic'].append(crit(torch.unsqueeze(bicub, 0), torch.unsqueeze(target, 0)))



    def test(self):
        netG = with_center(input_nc=1, output_nc=2, num_downs=6, scale_factor=UPSCALE_FACTOR)
        netG.load_state_dict(torch.load(self.params_path))

        print('# generator parameters:', sum(param.numel() for param in netG.parameters()))

        generator_criterion = GeneratorLossESRGAN()

        if torch.cuda.is_available():
            netG.cuda()
            generator_criterion.cuda()
            print("Model on CUDA")

        results_dict = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}


        #print("folder_name",time_stamp)
        idata = {}
        ## Validation:
        netG.eval()

        with torch.no_grad():
            val_bar =tqdm(self.val_loader)
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

                batch_ssim = pytorch_ssim.ssim(results, val_target).item()
                val_results['ssims'] += batch_ssim * batch_size



                tmp_psnr=10 * torch.log10((val_target.max()**2) / (val_results['mse'] / val_results['batch_sizes']))
                tmp_ssim= val_results['ssims'] / val_results['batch_sizes']

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
                        self.save_result_for_img(self.test_dir, results[i], val_target[i], bicub_imgs[i], input_imgs[i], x[i], idata, i )
                    break

        data_frame = pd.DataFrame(data=idata)
        data_frame.to_csv( os.path.join(self.test_dir, 'validation data.csv'))



def parser_arg(args):
    parser = argparse.ArgumentParser(description='CSRGAN')
    parser.add_argument('--train_path',default='./data/6k_data/train')
    parser.add_argument('--test_path',default='./data/6k_data/test')
    parser.add_argument('--type_of_dataset', default="10_dogs")
    parser.add_argument('--fname', default="")
    parser.add_argument('--params_path', default= "params/6k_params/netG_epoch_20.pth")
    parser.add_argument('--batch_size', default= 16,type=int)
    return parser.parse_args(args)

def main(args=None):
    if args==None:
        args=sys.argv[1:]
    args=parser_arg(args)
    model=CSRGANTest(args)
    model.test()


if __name__== "__main__":
    main()






