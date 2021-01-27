#####################
#     training      #
#####################
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
import torchvision.transforms as transforms
import sys
import argparse

sys.path.append("./code")
from dataset_cv2 import Places_dataset, save_results_img_grid4
from Discriminator import Discriminator, Conditional_Discriminator
from Generator import Generator
from Generator_break_63_plus_1 import GeneratorBroken63
from Generator_feature_extractor import GeneratorFeatures
from Generator_RRDB import GeneratorRRDB
from loss import GeneratorLoss, GeneratorLossESRGAN
from constants import *

import pytorch_ssim
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

time_stamp = datetime.now().strftime("%b%d_%H_%M")


class CSRGANTrain():
    def __init__(self,args):
        self.epochs=args.epochs
        self.saveparams_freq_batch=args.saveparams_freq_batch
        self.saveimg_freq_step=args.saveimg_freq_step
        self.batch_size=args.batch_size
        self.lrG=args.lrG
        self.lrD=args.lrD
        self.train_path=args.train_path
        self.test_path=args.test_path
        self.type_of_dataset=args.type_of_dataset
        self.fname=args.fname
        self.init_paths()
        self.prepare_dataset()
        self.init_models(args.generator,args.discriminator)


    def init_paths(self):
        self.out_path_sr_grid = f'./training_results/{time_stamp}/SR_results/'
        self.out_path_params = f'./training_results/{time_stamp}/params/'
        self.out_path_data = f'./training_results/{time_stamp}/train_data/'

        if not os.path.exists(self.out_path_sr_grid):
            os.makedirs(self.out_path_sr_grid)
        if not os.path.exists(self.out_path_params):
            os.makedirs(self.out_path_params)
        if not os.path.exists(self.out_path_data):
            os.makedirs(self.out_path_data)

        self.log_dir_name = self.out_path_data
                
    def prepare_dataset(self):
        train_dataset = Places_dataset(data_path=self.train_path, indexrange=(1, 5380), fname=self.fname,
                                       type_of_dataset=self.type_of_dataset,
                                       transforms=transforms.Compose(
                                           [transforms.Normalize(mean=middle_mean, std=middle_std)]))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

        val_dataset = Places_dataset(data_path=self.test_path, indexrange=(1, 595), fname=self.fname,
                                     type_of_dataset=self.type_of_dataset, full_data=True,
                                     transforms=transforms.Compose(
                                         [transforms.Normalize(mean=middle_mean, std=middle_std)]))

        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        self.train_loader=train_loader
        self.val_loader=val_loader

    def save_results_img_grid(self,path, gen_result, target, fname='results_grid.png', tot_images=10, nrows=2):
        red_r = gen_result[:, 2:, :, :]
        blue_r = gen_result[:, 0:1, :, :]
        green_r = gen_result[:, 1:2, :, :]
        gen_result = torch.cat((red_r, green_r, blue_r), 1)
        red_t = target[:, 2:, :, :]
        blue_t = target[:, 0:1, :, :]
        green_t = target[:, 1:2, :, :]
        target = torch.cat((red_t, green_t, blue_t), 1)
        imgsize = target.shape[2]
        datatensor = torch.empty((tot_images * 2, 3, imgsize, imgsize))
        res = gen_result[:tot_images, :, :, :]
        tar = target[:tot_images, :, :, :]
        datatensor[0::2, :, :, :] = tar
        datatensor[1::2, :, :, :] = res
        grid_image = utils.make_grid(datatensor, nrow=nrows, padding=3)
        utils.save_image(grid_image, os.path.join(path, fname), padding=3)


    def write_to_tb(self,tb, epoch, results):
        tb.add_scalar('loss/generator', results['g_loss'][-1], epoch)
        tb.add_scalar('loss/discriminator', results['d_loss'][-1], epoch)
        tb.add_scalar('score/discriminator', results['g_score'][-1], epoch)
        tb.add_scalar('score/discriminator', results['d_score'][-1], epoch)
        tb.add_scalar('results/psnr', results['psnr'][-1], epoch)
        tb.add_scalar('results/ssim', results['ssim'][-1], epoch)

    def init_models(self,generator,discriminator):
        if generator=='Generator':
            self.netG = Generator(input_nc=1, output_nc=2, num_downs=6,scale_factor=UPSCALE_FACTOR)
        elif generator=='GeneratorBroken63':
            self.netG = GeneratorBroken63(input_nc=1, output_nc=63, num_downs=6,scale_factor=UPSCALE_FACTOR)
        elif generator=='GeneratorFeatures':
            self.netG = GeneratorFeatures(input_nc=1, output_nc=2, num_downs=6, scale_factor=UPSCALE_FACTOR)
        elif generator=='GeneratorRRDB':
            self.netG=GeneratorRRDB(input_nc=1, output_nc=2, num_downs=6,scale_factor=UPSCALE_FACTOR)
        else:
            print("Generator Not supported")
            exit(0)

        if discriminator=='Discriminator':
            self.netD = Discriminator()
            self.Conditional_Discriminator=False
        elif discriminator=='Conditional_Discriminator':
            self.netD=Conditional_Discriminator()
            self.Conditional_Discriminator=True
        else:
            print("Discriminator Not supported")
            exit(0)

    def train(self):
        tb = SummaryWriter(log_dir=self.log_dir_name)

        print('# generator parameters:', sum(param.numel() for param in self.netG.parameters()))

        print('# discriminator parameters:', sum(param.numel() for param in self.netD.parameters()))

        generator_criterion = GeneratorLossESRGAN()

        if torch.cuda.is_available():
            self.netG.cuda()
            self.netD.cuda()
            generator_criterion.cuda()
            print("Model on CUDA")

        optimizerG = optim.Adam(self.netG.parameters(), lr=self.lrG, betas=(0.9, 0.999))
        optimizerD = optim.Adam(self.netD.parameters(), lr=self.lrD, betas=(0.9, 0.999))

        results_dict = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

        for epoch in range(self.epochs):
            print("folder_name", time_stamp)
            train_bar = tqdm(self.train_loader)
            running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

            self.netG.train()
            self.netD.train()
            train_batch_num = -1
            for data, target in train_bar:
                data = Variable(data)
                target = Variable(target)
                train_batch_num += 1
                if data == None or target == None:
                    print("ERROR reading batch, skipping batch num:", train_batch_num)
                    continue
                # g_update_first = True
                batch_size = data.size(0)
                running_results['batch_sizes'] += batch_size
                if torch.cuda.is_available():
                    target = target.cuda()
                    data = data.cuda()

                gray_img = data[:, 0:1, :, :]
                results = self.netG(gray_img)

                ############################
                # (1) Update D network: maximize D(x)-1-D(G(z))
                ###########################

                self.netD.zero_grad()
                if self.Conditional_Discriminator:
                    real_out = self.netD(target,gray_img)
                    fake_out = self.netD(results,gray_img)
                else:
                    real_out = self.netD(target)
                    fake_out = self.netD(results)
                ### end

                # BCE
                ones = torch.ones_like(real_out)
                if torch.cuda.is_available():
                    ones = ones.cuda()
                real_loss = nn.BCELoss()(real_out, ones)
                fake_loss = nn.BCELoss()(fake_out, 0 * ones)

                d_loss = real_loss + fake_loss
                d_loss.backward(retain_graph=True)
                optimizerD.step()

                ############################
                # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
                ###########################
                self.netG.zero_grad()

                if self.Conditional_Discriminator:
                    fake_out = self.netD(results, gray_img)
                else:
                    fake_out = self.netD(results)
                perceptual_loss = generator_criterion(fake_out, results, target)

                # g_loss = perceptual_loss + fool_GAN_loss
                g_loss = perceptual_loss
                g_loss.backward()
                optimizerG.step()

                # loss for current batch before optimization
                running_results['g_loss'] += g_loss.item() * batch_size
                running_results['d_loss'] += d_loss.item() * batch_size
                running_results['d_score'] += real_out.mean().item() * batch_size
                running_results['g_score'] += fake_out.mean().item() * batch_size

                # print("the problem:",type(running_results['d_loss']) ,  type(running_results['batch_sizes']))

                losd = running_results['d_loss'] / running_results['batch_sizes']
                losg = running_results['g_loss'] / running_results['batch_sizes']
                scrd = running_results['d_score'] / running_results['batch_sizes']
                scrg = running_results['g_score'] / running_results['batch_sizes']
                desc = f'[{epoch}/{self.epochs}], Loss_D: {losd:.3f} Loss_G: {losg:.3f} D(x): {scrd:.3f} D(G(z)): {scrg:.3f}'
                train_bar.set_description(desc=desc)

                if train_batch_num % self.saveimg_freq_step == 0:
                    self.save_results_img_grid(self.out_path_sr_grid, results, target,
                                          fname=f'results_{epoch}_iter{train_batch_num}.png')

            ## Validation:
            self.netG.eval()

            with torch.no_grad():
                val_bar = tqdm(self.val_loader)
                val_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
                val_images = []
                val_batch_num = -1
                for val_data, val_target, bicub_imgs, input_imgs in val_bar:
                    val_batch_num += 1
                    if val_data == None or val_target == None:
                        print("ERROR reading batch, skipping batch num:", val_batch_num)
                        continue
                    batch_size = val_data.size(0)
                    val_results['batch_sizes'] += batch_size

                    if torch.cuda.is_available():
                        val_data = val_data.cuda()
                        val_target = val_target.cuda()
                    results = self.netG(val_data[:, 0:1, :, :])

                    batch_mse = ((results - val_target) ** 2).mean()
                    val_results['mse'] += batch_mse * batch_size

                    batch_ssim = pytorch_ssim.ssim(results, val_target).item()  # shay?ssim computation..
                    val_results['ssims'] += batch_ssim * batch_size

                    tmp_psnr = 10 * torch.log10(
                        (val_target.max() ** 2) / (val_results['mse'] / val_results['batch_sizes']))
                    tmp_ssim = val_results['ssims'] / val_results['batch_sizes']

                    # val_bar.set_description(desc=f"[Validation] PSNR:{val_results['psnr']:.4f} dB SSIM: {val_results['ssim']:.4f}")
                    val_bar.set_description(desc=f"[Validation] PSNR:{tmp_psnr:.4f} dB SSIM: {tmp_ssim:.4f}")

                    save_prev_res = results
                    save_prev_tar = val_target

                    if val_batch_num == 1:
                        save_results_img_grid4(self.out_path_sr_grid, results, val_target, bicub_imgs, input_imgs,
                                               fname=f'results_val_{epoch}_iter{train_batch_num}.png')

                # shay? psnr computation:
                val_results['psnr'] = 10 * torch.log10(
                    (val_target.max() ** 2) / (val_results['mse'] / val_results['batch_sizes']))
                val_results['ssim'] = val_results['ssims'] / val_results['batch_sizes']

                # val_bar.set_description(desc=f"[Validation] PSNR:{val_results['psnr']:.4f} dB SSIM: {val_results['ssim']:.4f}")

            # save loss\scores\psnr\ssim
            # continue
            results_dict['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
            results_dict['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
            results_dict['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
            results_dict['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
            results_dict['psnr'].append(val_results['psnr'])
            results_dict['ssim'].append(val_results['ssim'])

            self.write_to_tb(tb, epoch, results_dict)

            if epoch % self.saveparams_freq_batch == 0:  # and epoch != 0:
                torch.save(self.netG.state_dict(), os.path.join(self.out_path_params, f'netG_epoch_{epoch}.pth'))
                torch.save(self.netD.state_dict(), os.path.join(self.out_path_params, f'netD_epoch_{epoch}.pth'))

                data_frame = pd.DataFrame(data={'Loss_D': results_dict['d_loss'], 'Loss_G': results_dict['g_loss'],
                                                'Score_D': results_dict['d_score'],
                                                'Score_G': results_dict['g_score'], 'PSNR': results_dict['psnr'],
                                                'SSIM': results_dict['ssim']}, index=range(epoch + 1))
                data_frame.to_csv(os.path.join(self.out_path_data, 'SR_train_results.csv'), index_label='epoch')

        tb.close()







def parser_arg(args):
    parser = argparse.ArgumentParser(description='CSRGAN')
    parser.add_argument('--epochs',default=21,type=int)
    parser.add_argument('--saveparams_freq_batch', default=5,type=int)
    parser.add_argument('--saveimg_freq_step', default=100,type=int)
    parser.add_argument('--lrG', default=1e-4,type=int)
    parser.add_argument('--lrD', default=1e-5,type=int)
    parser.add_argument('--train_path',default='./data/6k_data/train')
    parser.add_argument('--test_path',default='./data/6k_data/test')
    parser.add_argument('--type_of_dataset', default="10_dogs")
    parser.add_argument('--fname', default="")
    parser.add_argument('--generator', default="GeneratorFeatures")
    parser.add_argument('--discriminator', default= "Discriminator")
    parser.add_argument('--batch_size', default= 16,type=int)
    return parser.parse_args(args)

def main(args=None):
    if args==None:
        args=sys.argv[1:]
    args=parser_arg(args)
    model=CSRGANTrain(args)
    model.train()


if __name__== "__main__":
    main()


