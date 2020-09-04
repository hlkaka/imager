'''
Script to run the model
'''
import os
import sys
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything

from segmentation_models_pytorch.encoders import get_preprocessing_fn
import albumentations as A
from torchvision import transforms
from datetime import datetime
import torch

sys.path.append('data/')
from CTDataSet import CTDicomSlices, DatasetManager
from CustomTransforms import Window
from CustomTransforms import Imagify

sys.path.append('models/')
from UNet_L import UNet
from UNet_mateuszbuda import UNet_m
from run_model import get_datasets

from torchsummary import summary

import glob
from torch.utils.data import DataLoader

# Assumes holdout was done
# Train/val/test split
# Create dataset
# Create dataloader
# Create model
# Run model

dataset = "../organized_dataset_2"
model_output_parent = "../model_runs"

val_frac = 0.112
test_frac = 0.112

train_list = "train.txt"
val_list = "val.txt"
test_list = "test.txt"
params_file = "params.txt" # where to save params for this run

backbone = 'resnet34'
encoder_weights = 'imagenet'

same_image_all_channels = False

WL = 50
WW = 200

img_size = 256

lr = 0.0001
freeze_backbone = False
freeze_n_layers = 8

cpu_batch_size = 2
gpu_batch_size = 64

n_epochs = 10

# Augmentations
rotate=30
translate=(0.2, 0.2)
scale=(0.8, 1.3)
shear=(7, 7)
gaussian_noise_std = 0.3

optimizer_params = {
        'factor': 0.5,
        'patience': 5, 
        'cooldown': 5, 
        'min_lr': 1e-6
}

def get_time():
    now = datetime.now()
    return now.strftime("%Y-%m-%d-%H:%M:%S")

def get_batch_size():
    # Setup trainer
    if torch.cuda.is_available():
        batch_size = gpu_batch_size    
    else:
        batch_size = cpu_batch_size

    return batch_size

def test_model(model, test_dl):
    # Setup trainer
    if torch.cuda.is_available():
        trainer = Trainer(gpus=1, precision=16)
    else:
        trainer = Trainer(gpus=0)

    trainer.test(model, test_dataloaders=test_dl)

def get_model(model_dir, batch_size):
    ckpt = glob.glob("{}/lightning_logs/version_0/checkpoints/epoch*.ckpt".format(model_dir))[0]

    m = UNet.load_from_checkpoint(ckpt)
    m.freeze()

    summary(m, (3, img_size, img_size), device='cpu')

    return m

if __name__ == '__main__':
    seed_everything(seed=45)

    model_dir = ""

    while not os.path.isdir(model_dir):
        model_dir = input("Enter model directory: ")

        # where to store model params
        model_dir = "{}/{}".format(model_output_parent, model_dir)
        
    datasets = get_datasets(_same_image_all_channels = False, model_dir=model_dir, new_ds_split = False,
                    train_list = "{}/train.txt".format(model_dir), val_list="{}/val.txt".format(model_dir), test_list="{}/test.txt".format(model_dir))
    batch_size = get_batch_size()

    # create model
    #model = UNet(datasets, backbone=backbone, encoder_weights=encoder_weights, batch_size=batch_size, lr=lr, classes=2)
    model = get_model(model_dir, batch_size)

    test_dl = DataLoader(datasets['test'], batch_size=batch_size, num_workers = 8, shuffle=False)

    # save params
    '''note = input("Enter title for this training run:")
    params = "note: {}\ndataset: {}\nbatch_size: {}\nbackbone: {}\nencoder_weights: {}\nWL: {}\nWW: {}\nimg_size: {}\nLR: {}\n".format(
        note, dataset, batch_size, backbone, encoder_weights, WL, WW, img_size, lr
    )
    params += "\n\n# AUGMENTATIONS\n\n rotation degrees: {}\ntranslate: {}\nscale: {}\nshear: {}\nGaussian noise: {}\nSame image all channels: {}".format(
        rotate, translate, scale, shear, gaussian_noise_std, same_image_all_channels
    )

    params += "\nOptimizer params: {}".format(optimizer_params)

    with open("{}/{}".format(model_dir, params_file), "w") as f:
        f.write(params)
    '''

    test_model(model, test_dl)