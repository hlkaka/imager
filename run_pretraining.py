'''
Script to run the model
'''
import os
import sys
from pytorch_lightning import Trainer, loggers as pl_loggers
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.plugins import DDPPlugin

from torchvision import transforms
from datetime import datetime

sys.path.append('.')
from data.CTDataSet import CTDicomSlices, CTDicomSlicesJigsaw
from data.CustomTransforms import Window, Imagify
from models.ResNet_jigsaw import ResnetJigsaw
from run_model import get_dl_workers
from models.UNet_L import UNet_no_val

from constants import Constants

from torchsummary import summary

# Assumes holdout was done
# Train/val/test split
# Create dataset
# Create dataloader
# Create model
# Run model

dataset = Constants.ct_only_cleaned_resized
model_output_parent = Constants.model_outputs

params_file = "params.txt" # where to save params for this run

backbone = 'resnet34'
encoder_weights = 'imagenet' # or None
loss = 'cross_entropy'

WL = 50
WW = 200
mean = 61.0249
std = 78.3195

lr = 0.0001

cpu_batch_size = 2
gpu_batch_size = 32

n_epochs = 10

in_channels = 3

pre_train = 'felz' # can be 'felz' or 'jigsaw'

optimizer_params = {
        'factor': 0.5,
        'patience': 3, 
        'cooldown': 0, 
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
    dataset = Constants.ct_only_cleaned_resized
    dcm_list = CTDicomSlicesJigsaw.generate_file_list(dataset,
        dicom_glob='/*/*/dicoms/*.dcm')

    prep = transforms.Compose([Window(WL, WW), Imagify(WL, WW)]) #, Normalize(mean, std)])

    if pre_train == 'jigsaw':
        ctds = CTDicomSlicesJigsaw(dcm_list, preprocessing=prep,
            return_tile_coords=True, perm_path=Constants.default_perms)
    elif pre_train == 'felz':
        ctds = CTDicomSlices(dcm_list, preprocessing=prep, n_surrounding=in_channels // 2)
    else:
        raise Exception('Invalid pre_train mode of "{}"'.format(pre_train))

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
        trainer = Trainer(gpus=Constants.n_gpus, accelerator='ddp_spawn', plugins=DDPPlugin(find_unused_parameters=False), precision=16, logger=tb_logger, default_root_dir=model_dir, max_epochs=n_epochs)
    else:
        trainer = Trainer(gpus=0, default_root_dir=model_dir, logger=tb_logger, max_epochs=n_epochs)

    trainer.fit(model)

def get_model(datasets, batch_size):
    if pre_train == 'jigsaw':
        m = ResnetJigsaw(datasets, backbone=backbone, 
            lr=lr, batch_size=batch_size, dl_workers=get_dl_workers())
        
        summary(m, (9, 64, 64), device='cpu')

    elif pre_train == 'felz':
        ds = {'train': datasets, 'val': None, 'test': None}
        m = UNet_no_val(ds, backbone=backbone, batch_size=batch_size, loss=loss,
                in_channels=in_channels, dl_workers=get_dl_workers(), encoder_weights=encoder_weights)

        summary(m, (256, 256, in_channels), device='cpu')

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

    params += "\ndataset: {}\nin_channels: {}\npre_train: {}\nencoder_weights: {}\nloss: {}".format(dataset, in_channels, pre_train, encoder_weights, loss)

    with open("{}/{}".format(model_dir, params_file), "w") as f:
        f.write(params)

    train_model(model, model_dir)
