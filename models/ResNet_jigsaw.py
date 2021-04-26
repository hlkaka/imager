import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from segmentation_models_pytorch.encoders import resnet
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
import kornia
from loss import DiceLoss2
from utils import Utils

class UNet(pl.LightningModule):

    def __init__(self, datasets, backbone, encoder_weights :str = None,
                 jigsaw_size :int = 9, activation :str = 'softmax', batch_size :int = 32,
                 lr = 0.0001, dl_workers = 8, WL :int = 50, WW :int = 200, gaussian_noise_std = 0,
                 degrees=0, translate=(0, 0), scale=(1, 1), shear=(0, 0), max_pix = 255,
                 mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], optimizer_params = None, in_channels=1):
        super().__init__()
        '''
        datasets: train, test, val datasets
        backbone: object of type resnet. for example, torchvision.models.resnet34
        '''

        self.backbone = backbone
        self.

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
        self.ra = kornia.augmentation.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.8, 1.3), shear=(7, 7))
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
        if self.in_channels == 1:    # Hack to limit to single input
            return self.smp_unet(x[:,1,:,:].unsqueeze(1))
        else:
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
        #tensorboard_logs = {'train_loss': loss}
        return {'loss': loss} #, 'log': tensorboard_logs}

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
