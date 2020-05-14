
#######################################################################################################################
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch import optim
import torch.nn.init as init
from torchvision import transforms as tf
import cv2
import numpy as np
from torch import optim

class Res_down(nn.Module):
    def __init__(self, channel_in, channel_out, scale=2):
        super(Res_down, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, channel_out // 2, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(channel_out // 2)
        self.conv2 = nn.Conv2d(channel_out // 2, channel_out, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(channel_out)

        self.conv3 = nn.Conv2d(channel_in, channel_out, 3, 1, 1)

        self.AvePool = nn.AvgPool2d(scale, scale)

    def forward(self, x):
        skip = self.conv3(self.AvePool(x))

        x = F.rrelu(self.BN1(self.conv1(x)))
        x = self.AvePool(x)
        x = self.BN2(self.conv2(x))

        x = F.rrelu(x + skip)
        return x


class Res_up(nn.Module):
    def __init__(self, channel_in, channel_out, scale=2):
        super(Res_up, self).__init__()

        self.conv1 = nn.ConvTranspose2d(channel_in, channel_out // 2, 3, 1, 1)
        self.BN1 = nn.BatchNorm2d(channel_out // 2)
        self.conv2 = nn.ConvTranspose2d(channel_out // 2, channel_out, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(channel_out)

        self.conv3 = nn.ConvTranspose2d(channel_in, channel_out, 3, 1, 1)

        self.UpNN = nn.Upsample(scale_factor=scale, mode="nearest")

    def forward(self, x):
        skip = self.conv3(self.UpNN(x))

        x = F.rrelu(self.BN1(self.conv1(x)))
        x = self.UpNN(x)
        x = self.BN2(self.conv2(x))

        x = F.rrelu(x + skip)
        return x

class Encoder(nn.Module):
    def __init__(self, channels, ch=224, z=2000):
        super(Encoder, self).__init__()
        self.conv1 = Res_down(channels, ch)
        self.conv2 = Res_down(ch, 2 * ch)
        self.conv3 = Res_down(2 * ch, 2 * ch)
        self.conv4 = Res_down(2 * ch, 2 * ch)
        self.conv5 = nn.Conv2d(2 * ch, 2 * ch, 4, 3)
        self.conv_mu = nn.Conv2d(2 * ch, z, 4, 1)
        self.conv_logvar = nn.Conv2d(2 * ch, z, 4, 1)

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, Train=True):
        #print('Enter encoder:', x.shape)
        x = self.conv1(x)
        #print('Conv1:',x.shape)
        x = self.conv2(x)
        #print('Conv2:', x.shape)
        x = self.conv3(x)
        #print('Conv3:', x.shape)
        x = self.conv4(x)
        #print('Conv4:', x.shape)
        x = self.conv5(x)
        #print('Conv5:', x.shape)

        if Train:
            mu = self.conv_mu(x)
            #print('Conv mu:', mu.shape)
            logvar = self.conv_logvar(x)
            #print('Conv logvar:', logvar.shape)
            x = self.sample(mu, logvar)
            #print('sample shape', x.shape)
        else:
            x = self.conv_mu(x)
            mu = None
            logvar = None
        return x, mu, logvar

class Decoder(nn.Module):
    def __init__(self, channels, ch=224, z=2000):
        super(Decoder, self).__init__()
        self.deconvmu = nn.ConvTranspose2d(z, 2 * ch, 4, 1)
        self.deconv5 = nn.ConvTranspose2d(2 * ch, 2 * ch, 5, 3)
        self.conv1 = Res_up(2 * ch, ch * 2)
        self.conv2 = Res_up(ch * 2, ch )
        #self.conv3 = Res_up(ch * 2, ch * 2)
        self.conv4 = Res_up(ch , ch)
        #self.conv5 = Res_up(ch, ch // 2)
        #self.conv6 = nn.ConvTranspose2d(ch // 2, channels, 3, 1, 1)
        self.conv6 = nn.ConvTranspose2d(ch, channels, 3, 1, 1)

    def forward(self, x):
        #print('Decoder init', x.shape)
        x = self.deconvmu(x)
        #print('Decoder convmu', x.shape)
        x = self.deconv5(x)
        #print('Decoder conv5', x.shape)
        x = self.conv1(x)
        #print('Decoder conv1', x.shape)
        x = self.conv1(x)
        #print('Decoder conv1', x.shape)
        x = self.conv2(x)
        #print('Decoder conv1', x.shape)

        x = self.conv4(x)
        #print('Decoder conv1', x.shape)

        x = self.conv6(x)
        #print('Decoder conv1', x.shape)

        return x


class VAE(nn.Module):
    def __init__(self, channel_in, z=2000):
        super(VAE, self).__init__()
        self.encoder = Encoder(channel_in, z=z)
        self.decoder = Decoder(channel_in, z=z)

    def forward(self, x, Train=True):
        encoding, mu, logvar = self.encoder(x, Train)
        recon = self.decoder(encoding)
        return recon, mu, logvar, encoding

class Discriminator(nn.Module):
    def __init__(self, z_dim=2000):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(224*224*1 + z_dim, 4000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(4000, 4000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(4000, 2000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(2000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1),

        )

    def forward(self, x, z):
        x = x.view(-1,224*224)
        x = torch.cat((x, z), 1)
        return self.net(x).squeeze()

def permute_dims(z):
    #z = z[:,:,0,0]
    assert z.dim() == 2
    B, _ = z.size()
    perm = torch.randperm(B).to('cuda')
    perm_z = z[perm]
    return perm_z



#########################################################################################################################

def Augmentation(r=(-10,10), t=(0.1,0.1),shear=10, scale=(0.85, 1.15) ,b=(0.9, 1.1),cval=0):
    transformation = tf.Compose([
        tf.Resize((224, 224)),
        tf.ColorJitter(brightness=b),
        tf.RandomRotation(r, fill=cval),
        tf.RandomHorizontalFlip(),
        tf.RandomAffine(0,translate=t, shear=shear, scale=scale, fillcolor=cval),
        #TransformShow(), # visualize transformed pic
        tf.ToTensor(), # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] i
    ])
    return transformation

# for visualizing the transformations: taken from https://stackoverflow.com/questions/55179282/display-examples-of-augmented-images-in-pytorch
def TransformShow(name="transformed", wait=100):
    def transform_show(img):
        cv2.imshow(name, np.array(img))
        cv2.waitKey(wait)
        return img
    return transform_show

########################################################################################################################

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

train_folder = '/home/firedragon/Desktop/PROJECTS/COVID19/DD2424_COVID19_Project/datasets/covid-chestxray-dataset/images0'

train_dataset = ImageFolder(root=train_folder, transform=tf.Compose([tf.Resize((224, 224)),tf.ToTensor()]))
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

#train_dataset = ImageFolder(root=train_folder, transform=Augmentation())
#train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

train_dataset_wb = ImageFolder(root=train_folder, transform=tf.Compose([tf.Resize((224, 224)),tf.ToTensor()]))
train_loader_wb = DataLoader(train_dataset, batch_size=1, shuffle=True)

torch.device('cuda')

########################################################################################################################

import os
import time
import numpy as np
#import matplotlib.pyplot as plt
#import pickle

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
#import pytorch_ssim as pyssim
from skimage.metrics import structural_similarity as ssim
from skimage import data, img_as_float
#import tensorflow as tf

D = Discriminator()#.to('cuda')
optim_D = optim.Adam(D.parameters(), lr=0.0001)

def ELBO(recon_x, x, mu, logvar, beta):
    encoder_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) #KLD
    MSE = nn.MSELoss()
    decoder_loss = MSE(recon_x, x)
    total = beta*encoder_loss + decoder_loss
    return total, encoder_loss, decoder_loss

def train(data_loader, model, optimizer, device_use, beta):
    model.train()
    train_loss = 0
    encoder_loss = 0
    decoder_loss = 0
    acc_ssim = 0

    for idx in range(len(data_loader)):


        torch.device('cuda')

        datapoint,_ = next(iter(data_loader))

        data_rs = datapoint[0, 0, :, :].reshape(1,1,224,224)
        print(type(data_rs))
        #data_rs.to('cuda')

        optimizer.zero_grad()
        recon_batch, mu, logvar, z = model.forward(data_rs)


        z = z.reshape(-1,2000)
        D_xz = D(data_rs, z)
        print('yes')
        z_perm = permute_dims(z)
        D_x_z = D(data_rs, z_perm)

        Info_xz = -(D_xz.mean() - (torch.exp(D_x_z - 1).mean()))

        im1 = img_as_float(recon_batch.detach().numpy())[0, 0, :, :]
        im2 = img_as_float(data_rs.detach().numpy())[0, 0, :, :]

        acc_ssim += ssim(im1, im2, data_range=im1.max() - im2.min())
        loss_total, loss_part1, loss_part2 = ELBO(recon_batch[0,0,:,:].reshape(1,-1),
                                                      data_rs[0,0,:,:].reshape(1,-1), mu, logvar,beta)

        loss_infoVAE = loss_total + 100*Info_xz
        loss_infoVAE.backward()

        train_loss += loss_total.item()
        encoder_loss += loss_part1.item()
        decoder_loss += loss_part2.item()

        print('')
        print('Datapoint: ', idx)
        print('Encoder Loss: ', encoder_loss)
        print('Decoder Loss: ', decoder_loss)
        print('Train Loss', train_loss)
        optimizer.step()

    return train_loss, encoder_loss, decoder_loss, acc_ssim/len(data_loader)

def reconstruction(original_image,model):
    '''
    :param original_image:
    :param model:
    :return:
    '''
    rec_image, mu, logvar, z = model.forward(original_image,Train=False)
    return rec_image

########################################################################################################################



Lower_Bound_total_train = []
Lower_Bound_encoder_train = []
Lower_Bound_decoder_train = []
Lower_Bound_total_val = []
Lower_Bound_encoder_val = []
Lower_Bound_decoder_val = []
Acc_ssim_train = []
Acc_ssim_val = []


num_epochs = 300


Epochs = [ epoch for epoch in range(1, num_epochs + 1) ]

#torch.device('cuda')

model = VAE(1)#.to(device='cuda')

#torch.device('cuda')

results_directory = '/home/firedragon/Desktop/PROJECTS/COVID19/Results'

for epoch in range(1, num_epochs + 1):
    print('')
    print('')
    print('Epoch', epoch)
    '''
    with torch.no_grad():
        sample_lat = torch.randn(1, 20)
        sample = model.decoder(sample_lat)
        save_image(sample.view(1,3,224,224),
                   str(results_directory) + '/Training_Output/CNNVAEsample_gen_while_train_' + str(epoch) + '.jpg')
    '''

    # Note: try other optimizers #
    # training part
    LB_train = train(data_loader=train_loader,
                model=model,
                device_use='cuda',
                beta = 30,
                optimizer=optim.Adam(model.parameters(),lr=0.0001))

    Lower_Bound_total_train.append(LB_train[0])
    Lower_Bound_encoder_train.append(LB_train[1])
    Lower_Bound_decoder_train.append(LB_train[2])
    Acc_ssim_train.append(LB_train[3])



# recheck reconstructions after training
for idx in range(len(train_loader_wb)):
    datapoint, _ = next(iter(train_loader_wb))
    imgs = datapoint[:, 0, :, :].reshape(1,1,224,224)
    #imgs.to('cuda')
    torch.device('cuda')
    reconst_img_train = reconstruction(imgs, model)
    save_image(reconst_img_train.view(1,1,224,224),
               str(results_directory) + '/Reconstruction/Training/CNNVAErec_TRAIN_' + str(idx) + '.jpg')


np.save(file=results_directory + '/Graphs/Lower_Bound_total_train.npy', arr=Lower_Bound_total_train,
        allow_pickle=True)
np.save(file=results_directory + '/Graphs/Lower_Bound_encoder_train.npy', arr=Lower_Bound_encoder_train,
        allow_pickle=True)
np.save(file=results_directory + '/Graphs/Lower_Bound_decoder_train.npy', arr=Lower_Bound_decoder_train,
        allow_pickle=True)

np.save(file=results_directory + '/Graphs/Lower_Bound_total_val.npy', arr=Lower_Bound_total_val,
        allow_pickle=True)
np.save(file=results_directory + '/Graphs/Lower_Bound_encoder_val.npy', arr=Lower_Bound_encoder_val,
        allow_pickle=True)
np.save(file=results_directory + '/Graphs/Lower_Bound_decoder_val.npy', arr=Lower_Bound_decoder_val,
        allow_pickle=True)

np.save(file=results_directory + '/Graphs/Acc_ssim_train.npy', arr=Acc_ssim_train, allow_pickle=True)
np.save(file=results_directory + '/Graphs/Acc_ssim_val.npy', arr=Acc_ssim_val, allow_pickle=True)


'''
# outputs from training
plt.figure()
plt.plot(Epochs, Lower_Bound_total_train, label='Total')
plt.plot(Epochs, Lower_Bound_encoder_train, label='Encoder')
plt.plot(Epochs, Lower_Bound_decoder_train, label='Decoder')
plt.yscale('log')
plt.legend()
plt.title('Lower Bound detail: Train set')
plt.xlabel('Epochs')
plt.savefig(args.results_directory+'/Graphs/CNNVAEloss_train.jpg')

# outputs from validation
plt.figure()
plt.plot(Epochs, Lower_Bound_total_val, label='Total')
plt.plot(Epochs, Lower_Bound_encoder_val, label='Encoder')
plt.plot(Epochs, Lower_Bound_decoder_val, label='Decoder')
plt.yscale('log')
plt.legend()
plt.title('Lower Bound detail: Validation set')
plt.xlabel('Epochs')
plt.savefig(args.results_directory+'/Graphs/CNNVAEloss_validation.jpg')

#
plt.figure()
plt.plot(Epochs, Lower_Bound_total_train, label='Train')
plt.plot(Epochs, Lower_Bound_total_val, label='Validation')
plt.yscale('log')
plt.legend()
plt.title('Lower Bound')
plt.xlabel('Epochs')
plt.savefig(args.results_directory+'/Graphs/CNNVAEloss_train_plus_val.jpg')

plt.figure()
plt.plot(Epochs, Acc_ssim_train, label='Train')
plt.plot(Epochs, Acc_ssim_val, label='Validation')
plt.yscale('linear')
plt.legend()
plt.title('Structural Similarity Index')
plt.xlabel('Epochs')
plt.savefig(args.results_directory+'/Graphs/CNNVAEssim.jpg')
'''





