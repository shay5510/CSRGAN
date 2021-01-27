import torch
from torch import nn
from torchvision.models.vgg import vgg19,vgg16
from torchvision import transforms
from dataset_cv2 import bgr_to_rgb

imagenet_mean=[0.485,0.456,0.406]
imagenet_std=[0.229,0.224,0.225]


def bgr_to_rgb_imagenet_normalized(tensor_bgr):
    r=(tensor_bgr[:,2:,:,:]-0.485)/0.229
    g=(tensor_bgr[:,1:2,:,:]-0.456)/0.224
    b=(tensor_bgr[:,0:1,:,:]-0.406)/0.225
    return torch.cat((r,g,b),1)




class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        #adversarial_loss = torch.mean(1 - out_labels)
        adversarial_loss=-1*torch.mean(torch.log(out_labels))
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        print("mse loss:", image_loss) 
        print("prec loss:", perception_loss)
        print("adv loss:", adversarial_loss)
        return image_loss + 0.001 * adversarial_loss + 0.08 * perception_loss + 2e-8*tv_loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class GeneratorLossESRGAN(nn.Module):
    def __init__(self):
        super(GeneratorLossESRGAN, self).__init__()
        vgg = vgg19(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:36]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.L1_loss=nn.L1Loss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        #adversarial_loss = torch.mean(1 - out_labels)
        adversarial_loss=-1*torch.mean(torch.log(out_labels))
        #out_images_VGG=transforms.ToTensor()(out_images)
        #out_images_VGG=transforms.Normalize(mean=imagenet_mean,std=imagenet_std)(out_images_VGG)
        #normalize=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=imagenet_mean,std=imagenet_std)])
        
        #imagenet preparation
        out_images_VGG=bgr_to_rgb_imagenet_normalized(out_images)
        target_images_VGG=bgr_to_rgb_imagenet_normalized(target_images)
        
        # bgr to rgb
        out_images=bgr_to_rgb(out_images)
        target_images=bgr_to_rgb(target_images)

        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images_VGG), self.loss_network(target_images_VGG))
        # Image Loss 
        image_loss =self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss =( self.tv_loss(out_images))
        # L1_loss
        image_L1_loss=self.L1_loss(out_images, target_images)

        print("mse loss:", image_loss)
        print("prec loss:", perception_loss)
        print("adv loss:", adversarial_loss)

        #return image_loss + 0.005 * adversarial_loss + perception_loss
        #0.006 preception_loss
        return image_loss +  0.001*adversarial_loss+0.006*perception_loss +2e-8*tv_loss



if __name__ == "__main__":
    #g_loss = GeneratorLoss()
    g_loss = GeneratorLossESRGAN()
    print(g_loss)
