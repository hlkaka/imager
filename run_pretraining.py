'''
Script to run the model
'''
import os
import sys
from pytorch_lightning import Trainer, loggers as pl_loggers
from pytorch_lightning.utilities.seed import seed_everything

from torchvision import transforms
from datetime import datetime

sys.path.append('.')
from data.CTDataSet import CTDicomSlicesJigsaw
from data.CustomTransforms import Window, Imagify, Normalize
from models.ResNet_jigsaw import ResnetJigsaw
from run_model import get_dl_workers

from constants import Constants

from torchsummary import summary

# Assumes holdout was done
# Train/val/test split
# Create dataset
# Create dataloader
# Create model
# Run model

dataset = Constants.ct_only_cleaned
model_output_parent = Constants.model_outputs

val_frac = 0.112
test_frac = 0.112

params_file = "params.txt" # where to save params for this run

backbone = 'resnet34'

WL = 50
WW = 200
mean = 61.0249
std = 78.3195

lr = 0.0001

cpu_batch_size = 2
gpu_batch_size = 32

n_epochs = 10

in_channels = 1  # Hack in UNet_L at the moment to make this work

# Augmentations
rotate=15
translate=(0.1, 0.1)
scale=(0.9, 1.1)

optimizer_params = {
        'factor': 0.5,
        'patience': 5, 
        'cooldown': 5, 
        'min_lr': 1e-6
}

def get_time():
    now = datetime.now()
    return now.strftime("%Y-%m-%d-%H:%M:%S")

def get_dataset():
    '''
    Builds the necessary datasets
    '''
    # create ds
    dataset = Constants.ct_only_cleaned
    dcm_list = CTDicomSlicesJigsaw.generate_file_list(dataset,
        dicom_glob='/*/*/dicoms/*.dcm')

    prep = transforms.Compose([Window(WL, WW), Imagify(WL, WW)]) #, Normalize(mean, std)])
    ctds = CTDicomSlicesJigsaw(dcm_list, preprocessing=prep, trim_edges=True,
            return_tile_coords=True, perm_path=Constants.default_perms)

    return ctds

def get_batch_size():
    # Setup trainer
    if Constants.n_gpus != 0:
        batch_size = gpu_batch_size    
    else:
        batch_size = cpu_batch_size

    return batch_size

def train_model(model, model_dir):
    # Setup trainer
    tb_logger = pl_loggers.TensorBoardLogger('{}/logs/'.format(model_dir))
    if Constants.n_gpus != 0:
        trainer = Trainer(gpus=Constants.n_gpus, precision=16, logger=tb_logger, default_root_dir=model_dir, max_epochs=n_epochs)
    else:
        trainer = Trainer(gpus=0, default_root_dir=model_dir, logger=tb_logger, max_epochs=n_epochs)

    trainer.fit(model)

def get_model(datasets, batch_size):
    m = ResnetJigsaw(datasets, backbone=backbone, 
        lr=lr, batch_size=batch_size, dl_workers=get_dl_workers)

    summary(m, (9, 64, 64), device='cpu')

    return m

if __name__ == '__main__':
    seed_everything(seed=45)
    
    # where to store model params
    model_dir = "{}/pretrain-{}".format(model_output_parent, get_time())
    os.makedirs(model_dir, exist_ok=True)
    
    dataset = get_dataset()
    batch_size = get_batch_size()

    # create model
    #model = UNet(datasets, backbone=backbone, encoder_weights=encoder_weights, batch_size=batch_size, lr=lr, classes=2)
    model = get_model(dataset, batch_size)

    # save params
    note = input("Enter title for this training run:")
    params = "note: {}\ndataset: {}\nbatch_size: {}\nbackbone: {}\nWL: {}\nWW: {}\nmean: {}\nstd: {}\ntile_size: 64\nLR: {}\n".format(
        note, dataset, batch_size, backbone, WL, WW, mean, std, lr
    )

    params += "\nOptimizer params: {}".format(optimizer_params)

    with open("{}/{}".format(model_dir, params_file), "w") as f:
        f.write(params)

    train_model(model, model_dir)
