import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import sys
sys.path.append('.')
from data.CTDataSet import jigsaw_training_collate
import torchsort

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
        '''
        Converts tiles to a puzzle image
        Also replicates across given number of channels if provided
        '''
        # tiles is [batch_size, tile_n, tile H, tile W]
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

        # puzzle is [batch_size, C, puzzle H, puzzle W]
        return puzzle # replicates over the given number of channels

    def forward(self, x):
        # x is [batch_size, tile_n, tile H, tile W]
        x = ResnetJigsaw.tiles_to_image(x, device=self.device, in_channels=self.in_channels)
        
        # here, x is [batch_size, C, puzzle H, puzzle W]
        x = self.resnet(x)

        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        
        y_hat = self(images)
        loss = self.loss(y_hat, labels)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        
        y_hat = self(images)
        loss = self.loss(y_hat, labels)
        
        self.log('validation_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def train_dataloader(self):
        return DataLoader(self.datasets['train'], persistent_workers=True, batch_size=self.batch_size, num_workers = self.dl_workers,
                          shuffle=True, collate_fn=jigsaw_training_collate)

    def val_dataloader(self):
        return DataLoader(self.datasets['val'], persistent_workers=True, batch_size=self.batch_size, num_workers = self.dl_workers,
                          shuffle=False, collate_fn=jigsaw_training_collate)

    
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

class ResnetJigsaw_Ennead(ResnetJigsaw):
    def __init__(self, datasets, backbone = 'resnet34',
                 batch_size :int = 32, lr = 0.0001, dl_workers = 8,
                 optimizer_params = None, fc_size = 512,
                 num_permutations = 1000, in_channels = 1, pretrained=False):

        super().__init__(datasets, backbone=backbone, batch_size=batch_size, lr=lr, dl_workers=dl_workers,
                optimizer_params=optimizer_params, num_permutations=num_permutations, in_channels=in_channels, pretrained=pretrained)

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(num_ftrs, fc_size)

        self.puzzle_pieces = 9  # currently hardcoded. may be changed in future

        # fc7 is first context layer - same name convention as paper
        self.fc7 = torch.nn.Linear(fc_size * self.puzzle_pieces, 4096)
        self.fc8 = torch.nn.Linear(4096, num_permutations)

    def forward(self, x):
        # x is [batch_size, tile_n, tile H, tile W]
        # replicate across the number of channels to make x = [batch_size, tile_n, C, tile H, tile W]
        x = x.unsqueeze(2).repeat(1, 1, self.in_channels, 1, 1)

        all_outs = []

        for i in range(self.puzzle_pieces):
            all_outs.append(self.resnet(x[:, i]))  # pass each tile independently

        # each item in all_outs has dim [batch_size, fc_size]
        # concatenate to [batch_size, fc_size * self.puzzle_pieces]

        x = torch.cat(all_outs, dim=1)
        x = self.fc7(x)
        x = self.fc8(x)

        return x

class ResnetJigsawSR(ResnetJigsaw):
    def __init__(self, datasets, backbone = 'resnet34',
                 batch_size :int = 32, lr = 0.0001, dl_workers = 8,
                 optimizer_params = None,
                 num_permutations = 1000, in_channels = 1, pretrained=False):

        super().__init__(datasets, backbone=backbone, batch_size=batch_size, lr=lr,
                dl_workers=dl_workers, optimizer_params=optimizer_params, num_permutations=num_permutations,
                in_channels=in_channels, pretrained=pretrained)

        puzzle_size = 9

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Linear(num_ftrs, puzzle_size)
        self.loss_ = torch.nn.MSELoss(reduction='mean')

    def loss(self, y_hat, y):
        return self.loss_(y_hat, y.float())

    def forward(self, x):
        x = super().forward(x)
        return torchsort.soft_sort(x.float(), regularization_strength=1.0) # soft_sort not implemented for half-precision