import torch
from torch import nn
from torchvision.models.vgg import vgg19,vgg16
from torchvision import transforms
from code.dataset_cv2 import bgr_to_rgb
from code.loss import TVLoss,bgr_to_rgb_imagenet_normalized, GeneratorLossESRGAN
imagenet_mean=[0.485,0.456,0.406]
imagenet_std=[0.229,0.224,0.225]


class GeneratorLossESRGAN_comuted_advLoss(nn.Module):
    def __init__(self):
        super(GeneratorLossESRGAN_comuted_advLoss, self).__init__()
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
        # old line : adversarial_loss=-1*torch.mean(torch.log(out_labels))
        adversarial_loss = out_labels
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

        return image_loss +  0.001*adversarial_loss +0.006*perception_loss+2e-8*tv_loss



if __name__ == "__main__":
    #g_loss = GeneratorLoss()
    g_loss = GeneratorLossESRGAN()
    print(g_loss)

