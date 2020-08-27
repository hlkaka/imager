import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
import kornia
from loss import DiceLoss2
from utils import Utils

class UNet(pl.LightningModule):

    def __init__(self, datasets, backbone :str = 'resnet34', encoder_weights :str = 'imagenet',
                 classes :int = 2, activation :str = 'softmax', batch_size :int = 32, class_weights :list = [1, 5.725],
                 lr = 0.0001, dl_workers = 8, WL :int = 50, WW :int = 200, gaussian_noise_std = 0,
                 degrees=0, translate=(0, 0), scale=(1, 1), shear=(0, 0)):
        super().__init__()

        self.smp_unet = smp.Unet(backbone, encoder_weights = encoder_weights, classes = classes, activation = activation)
        self.datasets = datasets
        self.batch_size = batch_size
        self.class_weights = class_weights
        self.lr = lr
        self.dl_workers = dl_workers

        self.loss = DiceLoss2()
        self.WW = WW
        self.WL = WL
        self.gaussian_noise_std = gaussian_noise_std

        # Augmentations
        self.ra = kornia.augmentation.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.8, 1.3), shear=(7, 7))
        self.rf = kornia.augmentation.RandomHorizontalFlip()

    def forward(self, x):
        #x = x.permute(0, 3, 1, 2)
        # Assume batch is of shape (B, C, H, W)
        return self.smp_unet(x)

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
        return torch.optim.Adam(params = self.smp_unet.parameters(), lr=self.lr)
