import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            *self.conv_block(in_size=64, out_size=64, stride=2),

            *self.conv_block(in_size=64, out_size=128, stride=1),

            *self.conv_block(in_size=128, out_size=128, stride=2),

            *self.conv_block(in_size=128, out_size=256, stride=1),

            *self.conv_block(in_size=256, out_size=256, stride=2),

            *self.conv_block(in_size=256, out_size=512, stride=1),

            *self.conv_block(in_size=512, out_size=512, stride=2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        res = self.net(x) #
        res = res.view(batch_size)
        return torch.sigmoid(res)

    def conv_block(self, in_size=3, out_size=64, stride=1):
      block_list = [nn.Conv2d(in_size, out_size, kernel_size=3, stride=stride, padding=1),
                    nn.BatchNorm2d(out_size),
                    nn.LeakyReLU(0.2),]
      return block_list


class Conditional_Discriminator(nn.Module):
    def __init__(self):
        super(Conditional_Discriminator, self).__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            *self.conv_block(in_size=64, out_size=64, stride=2),

            *self.conv_block(in_size=64, out_size=128, stride=1),

            *self.conv_block(in_size=128, out_size=128, stride=2),

            *self.conv_block(in_size=128, out_size=255, stride=1)
        )
        self.net2 = nn.Sequential(

            *self.conv_block(in_size=256, out_size=256, stride=2),

            *self.conv_block(in_size=256, out_size=512, stride=1),

            *self.conv_block(in_size=512, out_size=512, stride=2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x, input_data):
        batch_size = x.size(0)
        res = self.net1(x)
        res = torch.cat([res,input_data], dim=1)
        res = self.net2(res)
        res = res.view(batch_size)
        return torch.sigmoid(res)

    def conv_block(self, in_size=3, out_size=64, stride=1):
      block_list = [nn.Conv2d(in_size, out_size, kernel_size=3, stride=stride, padding=1),
                    nn.BatchNorm2d(out_size),
                    nn.LeakyReLU(0.2),]
      return block_list
