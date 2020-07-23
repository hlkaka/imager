import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader

class UNet(pl.LightningModule):

    def __init__(self, datasets, backbone :str = 'resnet34', encoder_weights :str = 'imagenet',
                 classes :int = 2, activation :str = 'softmax', batch_size :int = 32, class_weights :list = [0.1487, 0.8513],
                 lr = 0.0001):
        super().__init__()

        self.smp_unet = smp.Unet(backbone, encoder_weights = encoder_weights, classes = classes, activation = activation)
        self.pre_processing_fn = get_preprocessing_fn(backbone, encoder_weights)
        self.datasets = datasets
        self.batch_size = batch_size
        self.class_weights = torch.tensor(class_weights)
        self.lr = lr

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        return self.smp_unet(x)

    def dice_metric(self, y_hat, masks):
        '''
        Calculates DICE on a single input
        This is because it is always reducing when given a batch, even when set to reduce = 'none'
        ?bug
        '''
        batch_size = y_hat.shape[0]
        dice = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

        for i in range(batch_size):
            y = y_hat[i].unsqueeze(0)
            mask = masks[i].unsqueeze(0)

            dice[i] = pl.metrics.functional.dice_score(y, mask)

        return dice

    def training_step(self, batch, batch_idx):
        images, masks, _, _ = batch

        y_hat = self(images)
        loss = F.cross_entropy(y_hat, masks, weight=self.class_weights)

        dice = self.dice_metric(y_hat, masks)

        # Logs
        #tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'dice': dice} #, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        images, masks, _, _ = batch

        y_hat = self(images)
        loss = F.cross_entropy(y_hat, masks, weight=self.class_weights)

        dice = self.dice_metric(y_hat, masks)
        
        # Logs
        #tensorboard_logs = {'val_loss': loss}
        return {'val_loss': loss, 'val_dice': dice} #, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_dice_mean = torch.stack([x['val_dice'] for x in outputs]).mean()
        print("Mean validation DICE: {}".format(val_dice_mean))

        return {'val_loss': val_loss_mean, 'val_dice': val_dice_mean}

    def test_step(self, batch, batch_nb):
        images, masks, _, _ = batch

        y_hat = self(images)
        loss = F.cross_entropy(y_hat, masks, weight=self.class_weights)

        dice = self.dice_metric(y_hat, masks)
        
        # Logs
        #tensorboard_logs = {'val_loss': loss}
        return {'test_loss': loss, 'test_dice': dice} #, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_dice_mean = torch.stack([x['test_dice'] for x in outputs]).mean()
        print("Mean test DICE: {}".format(test_dice_mean))

        return {'test_loss': test_loss_mean, 'test_dice': test_dice_mean}

    def train_dataloader(self):
        return DataLoader(self.datasets['train'], batch_size=self.batch_size, num_workers = 4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.datasets['val'], batch_size=self.batch_size, num_workers = 4, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.datasets['test'], batch_size=self.batch_size, num_workers = 4, shuffle=False)

    def configure_optimizers(self):
        return torch.optim.Adam(params = self.smp_unet.parameters(), lr=self.lr)