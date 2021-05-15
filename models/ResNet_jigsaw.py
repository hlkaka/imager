import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import sys
sys.path.append('.')
from data.CTDataSet import jigsaw_training_collate

# For permutation generation with maximal Hamming distance
# https://github.com/bbrattoli/JigsawPuzzlePytorch

class ResnetJigsaw(pl.LightningModule):

    def __init__(self, datasets, backbone = 'resnet34',
                 batch_size :int = 32, lr = 0.0001, dl_workers = 8,
                 optimizer_params = None,
                 num_permutations = 1000):

        super().__init__()
        '''
        datasets: train, test, val datasets
        resnet: object of type resnet. for example, torchvision.models.resnet34
        '''
        self.datasets = datasets
        self.resnet = torch.hub.load('pytorch/vision', backbone, pretrained=False)
        self.resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet.fc = torch.nn.Linear(512, 1000)
        #self.set_single_channel()

        self.batch_size = batch_size
        self.lr = lr
        self.dl_workers = dl_workers

        self.optimizer_params = optimizer_params

        self.loss = nn.CrossEntropyLoss()

    def set_single_channel(self):
        # ResNet normally takes 3 channels
        # This changes the first convolution layer to the correct number (usually just 1 channel)
        for module in self.resnet.modules():
            if isinstance(module, nn.Conv2d):
                target_module = module
                break
        
        target_module.in_channels = 1

    @classmethod
    def tiles_to_image(cls, tiles, device = torch.device('cpu')):
        tile_size = int(tiles.shape[2])
        snjp = int(tiles.shape[1] ** 0.5)
        batch_size = int(tiles.shape[0])

        puzzle = torch.empty((batch_size, tile_size * snjp, tile_size * snjp), device=device)

        for i in range(snjp):
            for j in range(snjp):
                tl = [i * tile_size, j * tile_size]           # top left
                br = [tl[0] + tile_size, tl[1] + tile_size]   # bottom right

                puzzle[:, tl[0]:br[0], tl[1]:br[1]] = tiles[:, i * snjp + j]

        return puzzle     

    def forward(self, x):
        # [batch, tile# [0-8], H, W]
        #out = torch.empty(self.ResNet_out_dim, device=self.device, dtype=x.dtype)
        #for i in range(self.jigsaw_size):
            # Concatenating all the tiles
        #    out[i:i+self.ResNet_out_dim] = self.resnet(x[:,i,:,:])
        x = ResnetJigsaw.tiles_to_image(x, device=self.device).unsqueeze(1)
        
        x = self.resnet(x)

        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        
        y_hat = self(images)
        loss = self.loss(y_hat, labels)

        # Logs
        #tensorboard_logs = {'train_loss': loss}
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return {'train_loss': loss}

    def train_epoch_end(self, outputs):
        train_loss_mean = torch.stack([x['train_loss'] for x in outputs]).mean()
        d = {'train_loss': train_loss_mean}
        print(d)
        return d

    def validation_step(self, batch, batch_nb):
        images, labels = batch

        y_hat = self(images)

        # loss dim is [batch, 1, img_x, img_y]
        # need to get rid of the second dimension so
        # size matches with mask
        loss = self.loss(y_hat, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # Logs
        #tensorboard_logs = {'val_loss': loss}
        return {'val_loss': loss} #, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        d = {'val_loss': val_loss_mean}
        print(d)
        return d

    def test_step(self, batch, batch_nb):
        images, labels = batch

        y_hat = self(images)

        # loss dim is [batch, 1, img_x, img_y]
        # need to get rid of the second dimension so
        # size matches with mask
        loss = self.loss(y_hat, labels)

        # Logs
        #tensorboard_logs = {'val_loss': loss}
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'test_loss': loss} #, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()

        return {'test_loss': test_loss_mean}

    def train_dataloader(self):
        return DataLoader(self.datasets, batch_size=self.batch_size, num_workers = self.dl_workers,
                          shuffle=True, collate_fn=jigsaw_training_collate)

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params = self.parameters(), lr=self.lr)
        if self.optimizer_params is None:
            pass
            #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        else:
            factor = self.optimizer_params['factor']
            patience = self.optimizer_params['patience']
            cooldown = self.optimizer_params['cooldown']
            min_lr = self.optimizer_params['min_lr']
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience, cooldown=cooldown, min_lr=min_lr)

        return [optimizer] #, [scheduler]
    