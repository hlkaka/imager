import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import kornia

import sys
sys.path.append('.')

from models.loss import DiceLoss2
from models.utils import Utils

class UNet(pl.LightningModule):

    def __init__(self, datasets, backbone :str = 'resnet34', encoder_weights :str = 'imagenet',
                 classes :int = 2, activation :str = 'softmax', batch_size :int = 32,
                 lr = 0.0001, dl_workers = 8, WL :int = 50, WW :int = 200, gaussian_noise_std = 0,
                 degrees=0, translate=(0, 0), scale=(1, 1), shear=(0, 0), max_pix = 255,
                 mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], optimizer_params = None, in_channels=3):
        super().__init__()

        self.smp_unet = smp.Unet(backbone, encoder_weights = encoder_weights, classes = classes, activation = activation, in_channels=in_channels)
        self.datasets = datasets
        self.batch_size = batch_size
        self.lr = lr
        self.dl_workers = dl_workers

        self.loss = DiceLoss2()
        self.WW = WW
        self.WL = WL
        self.gaussian_noise_std = gaussian_noise_std

        # Augmentations
        self.ra = kornia.augmentation.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear)
        self.rf = kornia.augmentation.RandomHorizontalFlip()

        mean = torch.tensor(mean)
        mean = mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        std = torch.tensor(std)
        std = std.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        # This approach is necessary so lightning knows to move this tensor to appropriate device
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        self.max_pix = max_pix # pixel range is from 0 to this value

        self.optimizer_params = optimizer_params

        # Hack to keep track of input channels
        self.in_channels = in_channels

    def forward(self, x):
        #x = x.permute(0, 3, 1, 2)
        # Assume batch is of shape (B, C, H, W)
        return self.smp_unet(x)

    def training_step(self, batch, batch_idx):
        images, masks, _, _ = batch

        images, masks = Utils.preprocessing(images, masks, self.WL, self.WW)

        images, masks = Utils.do_train_augmentations(images, masks,
                self.gaussian_noise_std, self.device, self.ra, self.rf)

        images.div_(self.max_pix).sub_(self.mean).div_(self.std)

        y_hat = self(images)

        # loss dim is [batch, 1, img_x, img_y]
        # need to get rid of the second dimension so
        # size matches with mask
        loss = self.loss(y_hat[:,0,:,:], masks)

        # Logs
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        return loss #, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        images, masks, _, _ = batch

        images, masks = Utils.preprocessing(images, masks, self.WL, self.WW)
        images.div_(self.max_pix)
        images.sub_(self.mean).div_(self.std)
        y_hat = self(images)

        # loss dim is [batch, 1, img_x, img_y]
        # need to get rid of the second dimension so
        # size matches with mask
        loss = self.loss(y_hat[:,0,:,:], masks)

        # Logs

        return {'val_loss': loss} 
        #self.log('validation_loss', loss, on_step=True, on_epoch=True, sync_dist=True)

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        d = {'val_loss_mean': val_loss_mean}
        self.log('val_loss_mean', val_loss_mean, logger=True)
        
        return d

    def test_step(self, batch, batch_nb):
        images, masks, _, _ = batch

        images, masks = Utils.preprocessing(images, masks, self.WL, self.WW)

        images.div_(self.max_pix).sub_(self.mean).div_(self.std)

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
        self.log('test_loss_mean', test_loss_mean, logger=True)

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
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        else:
            factor = self.optimizer_params['factor']
            patience = self.optimizer_params['patience']
            cooldown = self.optimizer_params['cooldown']
            min_lr = self.optimizer_params['min_lr']
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=factor, patience=patience, cooldown=cooldown, min_lr=min_lr)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss_mean'
        }
        #[optimizer], [scheduler]
