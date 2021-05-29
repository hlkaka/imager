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
                 num_permutations = 1000, in_channels = 1, pretrained=False):

        super().__init__()
        '''
        datasets: train, test, val datasets
        resnet: object of type resnet. for example, torchvision.models.resnet34
        '''
        self.datasets = datasets
        self.resnet = torch.hub.load('pytorch/vision', backbone, pretrained=pretrained)
        if in_channels != 3:
            # If 3, keep the existing first layer as it might be pretrained
            self.resnet.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(num_ftrs, num_permutations)

        self.batch_size = batch_size
        self.lr = lr
        self.dl_workers = dl_workers

        self.optimizer_params = optimizer_params

        self.loss = nn.CrossEntropyLoss()
        self.in_channels = in_channels

    @classmethod
    def tiles_to_image(cls, tiles, device = torch.device('cpu'), in_channels = 1):
        tile_size = int(tiles.shape[2])
        snjp = int(tiles.shape[1] ** 0.5)
        batch_size = int(tiles.shape[0])

        puzzle = torch.empty((batch_size, tile_size * snjp, tile_size * snjp), device=device)

        for i in range(snjp):
            for j in range(snjp):
                tl = [i * tile_size, j * tile_size]           # top left
                br = [tl[0] + tile_size, tl[1] + tile_size]   # bottom right

                puzzle[:, tl[0]:br[0], tl[1]:br[1]] = tiles[:, i * snjp + j]

        # Add dimension for channels
        puzzle = puzzle.unsqueeze(1).repeat(1, in_channels, 1, 1)

        return puzzle # replicates over the given number of channels

    def forward(self, x):
        # TODO: copy the image to create multi-channel input
        x = ResnetJigsaw.tiles_to_image(x, device=self.device, in_channels=self.in_channels)
        
        x = self.resnet(x)

        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        
        y_hat = self(images)
        loss = self.loss(y_hat, labels)

        # Logs
        #tensorboard_logs = {'train_loss': loss}
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def train_dataloader(self):
        return DataLoader(self.datasets, persistent_workers=True, batch_size=self.batch_size, num_workers = self.dl_workers,
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
    