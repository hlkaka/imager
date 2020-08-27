'''
Alternative implementation of UNet_m from Torch Hub
https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py
'''

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torch.utils.data import DataLoader

import kornia

from loss import DiceLoss2

import sys
sys.path.append('data/')
from CustomTransforms import TorchFunctionalTransforms as TFT
from utils import Utils

class UNet_m(pl.LightningModule):

    def __init__(self, datasets, in_channels=3, out_channels=1,
                 init_features=32, lr=0.0001, batch_size = 32, dl_workers = 8, class_weights :list = [1, 5.725],
                 WL :int = 50, WW :int = 200, gaussian_noise_std = 0,
                 degrees=0, translate=(0, 0), scale=(1, 1), shear=(0, 0),
                 optimizer_params = None):
        super(UNet_m, self).__init__()

        features = init_features
        self.encoder1 = UNet_m._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet_m._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet_m._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet_m._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet_m._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet_m._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet_m._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet_m._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet_m._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

        # training params
        self.loss = DiceLoss2()
        self.lr = lr
        self.datasets = datasets
        self.batch_size = batch_size
        self.dl_workers = dl_workers
        self.class_weights = class_weights

        self.WW = WW
        self.WL = WL
        self.gaussian_noise_std = gaussian_noise_std

        self.optimizer_params = optimizer_params

        # Augmentations
        self.ra = kornia.augmentation.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.8, 1.3), shear=(7, 7))
        self.rf = kornia.augmentation.RandomHorizontalFlip()

    def forward(self, x):
        '''
        Expects input to be (B, C, H, W)
        Remember to call preprocessing before this function
        '''

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

    def training_step(self, batch, batch_idx):
        images, masks, _, _ = batch

        images, masks = Utils.preprocessing(images, masks, self.WL, self.WW)
        images, masks = Utils.do_train_augmentations(images, masks,
                self.gaussian_noise_std, self.device, self.ra, self.rf)

        y_hat = self(images)

        # loss dim is [batch, 1, img_x, img_y]
        # need to get rid of the second dimension so
        # size matches with mask
        loss = self.loss(y_hat[:,0,:,:], masks)

        # Logs
        #tensorboard_logs = {'train_loss': loss}
        return {'loss': loss} #, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        images, masks, _, _ = batch

        images, masks = Utils.preprocessing(images, masks, self.WL, self.WW)

        y_hat = self(images)

        # loss dim is [batch, 1, img_x, img_y]
        # need to get rid of the second dimension so
        # size matches with mask
        loss = self.loss(y_hat[:,0,:,:], masks)

        # Logs
        #tensorboard_logs = {'val_loss': loss}
        return {'val_loss': loss} #, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        d = {'val_loss': val_loss_mean}
        print(d)
        return d

    def test_step(self, batch, batch_nb):
        images, masks, _, _ = batch

        images, masks = Utils.preprocessing(images, masks, self.WL, self.WW)

        y_hat = self(images)

        # loss dim is [batch, 1, img_x, img_y]
        # need to get rid of the second dimension so
        # size matches with mask
        loss = self.loss(y_hat[:,0,:,:], masks)

        # Logs
        #tensorboard_logs = {'val_loss': loss}
        return {'test_loss': loss} #, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()

        return {'test_loss': test_loss_mean}

    def train_dataloader(self):
        return DataLoader(self.datasets['train'], batch_size=self.batch_size, num_workers = self.dl_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.datasets['val'], batch_size=self.batch_size, num_workers = self.dl_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.datasets['test'], batch_size=self.batch_size, num_workers = self.dl_workers, shuffle=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params = self.parameters(), lr=self.lr)
        if self.optimizer_params is None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        else:
            factor = self.optimizer_params['factor']
            patience = self.optimizer_params['patience']
            cooldown = self.optimizer_params['cooldown']
            min_lr = self.optimizer_params['min_lr']
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience, cooldown=cooldown, min_lr=min_lr)

        return [optimizer], [scheduler]