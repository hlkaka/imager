import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

# For permutation generation with maximal Hamming distance
# https://github.com/bbrattoli/JigsawPuzzlePytorch

class UNet(pl.LightningModule):

    def __init__(self, datasets, backbone = 'resnet34', encoder_weights :str = None,
                 jigsaw_size :int = 9, batch_size :int = 32, lr = 0.0001, dl_workers = 8,
                 optimizer_params = None, in_channels=1, ResNet_out_dim = 1000,
                 num_permutations = 1000):

        super().__init__()
        '''
        datasets: train, test, val datasets
        resnet: object of type resnet. for example, torchvision.models.resnet34
        '''
        self.datasets = datasets
        self.resnet = torch.hub.load('pytorch/vision', backbone, pretrained=False)
        self.set_single_channel()

        self.fc1 = nn.Linear(ResNet_out_dim * jigsaw_size, 4096)
        self.fc2 = nn.Linear(4096, num_permutations)

        self.batch_size = batch_size
        self.lr = lr
        self.dl_workers = dl_workers
        self.ResNet_out_dim = ResNet_out_dim
        self.jigsaw_size = jigsaw_size

        self.loss = nn.CrossEntropyLoss()

    def set_single_channel(self):
        # ResNet normally takes 3 channels
        # This changes the first convolution layer to the correct number (usually just 1 channel)
        for module in self.resnet.modules():
            if isinstance(module, nn.Conv2d):
                break
        
        module.in_channels = in_channels

    def forward(self, x):
        # 9 siamese ennead
        # [batch, tile# [0-8], H, W]
        out = torch.empty(self.ResNet_out_dim, device=self.device, dtype=x.dtype)
        for i in range(self.jigsaw_size):
            # Concatenating all the tiles
            out[i:i+self.ResNet_out_dim] = self.resnet(x[:,i,:,:])

        out = self.fc1(out)
        return self.fc2(out)

    def training_step(self, batch, batch_idx):
        images, perms = batch

        y_hat = self(images)

        # loss dim is [batch, 1, img_x, img_y]
        # need to get rid of the second dimension so
        # size matches with mask
        loss = self.loss(y_hat, perms)

        # Logs
        #tensorboard_logs = {'train_loss': loss}
        return {'loss': loss} #, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        images, perms = batch

        y_hat = self(images)

        # loss dim is [batch, 1, img_x, img_y]
        # need to get rid of the second dimension so
        # size matches with mask
        loss = self.loss(y_hat, perms)

        # Logs
        #tensorboard_logs = {'val_loss': loss}
        return {'val_loss': loss} #, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        d = {'val_loss': val_loss_mean}
        print(d)
        return d

    def test_step(self, batch, batch_nb):
        images, perms = batch

        y_hat = self(images)

        # loss dim is [batch, 1, img_x, img_y]
        # need to get rid of the second dimension so
        # size matches with mask
        loss = self.loss(y_hat, perms)

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
