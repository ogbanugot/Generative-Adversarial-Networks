#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 11:21:56 2018

@author: ogban ugot
"""
# =============================================================================
# Importing the libraries
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

batchSize = 64 
imageSize = 64 

ext = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

transform = transforms.Compose([transforms.Resize(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]) # We create a list of transformations (scaling, tensor conversion, normalization) to apply to the input images.

#dataset = dset.DatasetFolder(root='./data/L', loader='default_loader',extensions=ext, transform=transform)

dataset = dset.ImageFolder(root='./data2', transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2) # We use dataLoader to get the images of the training set batch by batch.


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class G(nn.Module):

    def __init__(self):
        super(G, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(False),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(False),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(False),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(False),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
            nn.Tanh()
        )

    def forward(self, input):
        output = self.main(input)
        return output

netG = G()
netG.apply(weights_init)


class D(nn.Module):

    def __init__(self):
        super(D, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias = False),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(64, 128, 4, 2, 1, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(128, 256, 4, 2, 1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(256, 512, 4, 2, 1, bias = False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(512, 1, 4, 1, 0, bias = False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1)

netD = D()
netD.apply(weights_init)

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999))


device = torch.device('cuda:0')
real_label = 1
fake_label = 0

for epoch in range(25):

    for i, data in enumerate(dataloader, 0):

        netD.zero_grad()
        real_cpu = Variable(data[0].to(device))
        batch_size = real_cpu.size(0)
        label = Variable(torch.full((batch_size,), real_label, device=device))

        output = netD(input)
        errD_real = criterion(output, label)                
        noise = Variable(torch.randn(batch_size, 100, 1, 1, device=device))
        fake = netG(noise)
        label = Variable(label.fill_(fake_label))        
        output = netD(fake.detach())
        errD_fake = criterion(output, label)
        
        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()

        netG.zero_grad()
        label = Variable(label.fill_(real_label))  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()
        

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 25, i, len(dataloader), errD.data[0], errG.data[0]))

        my_file=open("metrics.txt","a")
        my_file.write('%d; %d; %.4f; %.4f' % (epoch, i, errD.data[0], errG.data[0]) +'\n')
        my_file.close()

        
        if i % 100 == 0:
            vutils.save_image(real_cpu, '%s/real_samples.png' % "./results", nrow=8, normalize = True)
            fake = netG(noise)
            vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize = True)
	# do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % ("./checkpoint", epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % ("./checkpoint", epoch))
