from torchvision.models import resnet18
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import pdb

    

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        resnet18_model = resnet18(pretrained=True)      
        self.feature_extractor = nn.Sequential(*list(resnet18_model.children())[:-3])
        self.pooling = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)


        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, img):
        out = self.feature_extractor(img)
        out = self.pooling(out)
        out = out.view(out.size(0), -1)
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        return mu, logvar



class Generator(nn.Module):

    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        channels, self.h, self.w = img_shape

        self.conv1 = nn.Sequential(
            nn.Conv2d(3+latent_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
       
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        

        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        

        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        


        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )


        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        

        
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.ReLU(inplace=True)
            )   

        
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.ReLU(inplace=True)
            )

        
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.ReLU(inplace=True)
            )


        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(inplace=True)
            )

        
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(inplace=True)
            )

        
        self.deconv6 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True)
            )

        
        self.deconv7 = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
            )


    def forward(self, x, z):

        
        z = z.unsqueeze(dim=2).unsqueeze(dim=3)
        z = z.expand(z.size(0), z.size(1), x.size(2), x.size(3))
        x_with_z = torch.cat([x, z], dim=1)


        
        d1 = self.conv1(x_with_z)
        d2 = self.conv2(d1)
        d3 = self.conv3(d2)
        d4 = self.conv4(d3)
        d5 = self.conv5(d4)

        d7 = self.conv7(d5)

        u1 = self.deconv1(d7)
        u3 = self.deconv3(torch.cat([u1, d5], dim=1))
        u4 = self.deconv4(torch.cat([u3, d4], dim=1))
        u5 = self.deconv5(torch.cat([u4, d3], dim=1))
        u6 = self.deconv6(torch.cat([u5, d2], dim=1))
        out = self.deconv7(torch.cat([u6, d1], dim=1))

        return out




class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.d_1 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=2, padding=0, count_include_pad=False),
                                 nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                 nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                                 nn.InstanceNorm2d(64, affine=True),
                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                 nn.Conv2d(64, 128, kernel_size=4, stride=1, padding=1),
                                 nn.InstanceNorm2d(128, affine=True),
                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                 nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1),
                                 nn.ReLU(inplace=True)
                                 )


        self.d_2 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                 nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                                 nn.InstanceNorm2d(128, affine=True),
                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                 nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1),
                                 nn.InstanceNorm2d(256, affine=True),
                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                 nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1),
                                 nn.ReLU(inplace=True)
                                 )

    def forward(self, x):
        out_1 = self.d_1(x)
        out_2 = self.d_2(x)
        return out_1, out_2








