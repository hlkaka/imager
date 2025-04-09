'''
Script to run the model
'''
import os
import sys
from pytorch_lightning import Trainer, loggers as pl_loggers
#from lightning_fabric.utilities.seed import seed_everything
from pytorch_lightning.utilities.seed import seed_everything
#from pytorch_lightning.plugins import DDPPlugin

from torchvision import transforms
from datetime import datetime

sys.path.append('.')
from data.CTDataSet import CTDicomSlices, CTDicomSlicesJigsaw, DatasetManager
from data.CustomTransforms import Window, Imagify
from models.ResNet_jigsaw import ResnetJigsaw, ResnetJigsawSR, ResnetJigsaw_Ennead
from run_model import get_dl_workers
from models.UNet_L import UNet_no_val

from constants import Constants

from torchsummary import summary

from pytorch_lightning.callbacks import ModelCheckpoint

# Assumes holdout was done
# Train/val/test split
# Create dataset
# Create dataloader
# Create model
# Run model

dataset_dir = Constants.ct_only_cleaned # use ct_only_cleaned_resized for felz
model_output_parent = Constants.model_outputs

params_file = "params.txt" # where to save params for this run

backbone = 'resnet34'
encoder_weights = 'imagenet' # or None
loss = 'cross_entropy'

WL = 50
WW = 200
mean = 61.0249
std = 78.3195

lr = 0.00001

cpu_batch_size = 2
gpu_batch_size = 32

n_epochs = 50

in_channels = 3

pre_train = 'jigsaw_softrank' # can be 'felz', 'jigsaw_ennead', jigsaw_softrank' or 'jigsaw'
num_classes = 6 # for 'felz' pretraining only. 5 segments & 0 for background
num_shuffles = 1 # for jigsaw pretraining only. how many shuffles to return per image per epoch

n_perms = 100

optimizer_params = None #{
#        'factor': 0.5,
#        'patience': 2, 
#        'cooldown': 0, 
#        'min_lr': 1e-6
#}

def get_time():
    now = datetime.now()
    return now.strftime("%Y-%m-%d-%H:%M:%S")

def get_dataset(dataset, model_dir):
    '''
    Builds the necessary datasets
    '''
    # create ds
    #dcm_list = CTDicomSlicesJigsaw.generate_file_list(dataset,
    #    dicom_glob='/*/*/dicoms/*.dcm')

    prep = transforms.Compose([Window(WL, WW), Imagify(WL, WW)]) #, Normalize(mean, std)])

    if pre_train == 'jigsaw' or pre_train == 'jigsaw_ennead' or pre_train == 'jigsaw_softrank':
        if pre_train == 'jigsaw_softrank':
            n_perms = None

        dsm = DatasetManager.generate_train_val_test(dataset, val_frac=0.05, test_frac=0, pretrain_ds=True)
        if model_dir is not None:
            dsm.save_lists(model_dir)

        train_dicoms, val_dicoms, _ = dsm.get_dicoms()

        datasets = {}
        datasets['train'] = CTDicomSlicesJigsaw(train_dicoms, preprocessing=prep, return_tile_coords=True,
            perm_path=Constants.default_perms, n_shuffles_per_image=num_shuffles, num_perms=n_perms)
        
        datasets['val'] = CTDicomSlicesJigsaw(val_dicoms, preprocessing=prep, return_tile_coords=True,
            perm_path=Constants.default_perms, n_shuffles_per_image=num_shuffles, num_perms=n_perms)

        return datasets

    elif pre_train == 'felz':
        dcm_list = CTDicomSlicesJigsaw.generate_file_list(dataset,
                dicom_glob='/*/*/dicoms/*.dcm')

        # Felz masks were saved with foreground being 1,2,3,4 (instead of 255). mask_is_255 flag is CRITICAL
        ctds = CTDicomSlices(dcm_list, preprocessing=prep, n_surrounding=in_channels // 2, mask_is_255=False)
        
        return ctds

    else:
        raise Exception('Invalid pre_train mode of "{}"'.format(pre_train))

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

    chkpt1 = ModelCheckpoint(save_last=True) 
    chkpt2 = ModelCheckpoint(every_n_train_steps=10000) # save every 10000 steps

    if Constants.n_gpus != 0:
        trainer = Trainer(gpus=Constants.n_gpus, callbacks=[chkpt1, chkpt2], accelerator='ddp_spawn', plugins=DDPPlugin(find_unused_parameters=False), precision=16, logger=tb_logger, default_root_dir=model_dir, max_epochs=n_epochs)
    else:
        trainer = Trainer(gpus=0, default_root_dir=model_dir, logger=tb_logger, callbacks=[chkpt1, chkpt2], max_epochs=n_epochs)

    trainer.fit(model)

def get_model(datasets, batch_size):
    if pre_train == 'jigsaw':
        m = ResnetJigsaw(datasets, backbone=backbone, pretrained=(encoder_weights == 'imagenet'), optimizer_params=optimizer_params,
            lr=lr, batch_size=batch_size, dl_workers=get_dl_workers(), in_channels=in_channels, num_permutations=n_perms)
        
        summary(m, (9, 64, 64), device='cpu')

    elif pre_train == 'jigsaw_ennead':
        m = ResnetJigsaw_Ennead(datasets, backbone=backbone, pretrained=(encoder_weights == 'imagenet'), optimizer_params=optimizer_params,
            lr=lr, batch_size=batch_size, dl_workers=get_dl_workers(), in_channels=in_channels, num_permutations=n_perms)

        summary(m, (9, 64, 64), device='cpu')

    elif pre_train == 'jigsaw_softrank':
        m = ResnetJigsawSR(datasets, backbone=backbone, pretrained=(encoder_weights == 'imagenet'), optimizer_params=optimizer_params,
            lr=lr, batch_size=batch_size, dl_workers=get_dl_workers(), in_channels=in_channels)
        
        summary(m, (9, 64, 64), device='cpu')

    elif pre_train == 'felz':
        ds = {'train': datasets, 'val': None, 'test': None}
        m = UNet_no_val(ds, backbone=backbone, batch_size=batch_size, loss=loss, lr=lr,
                in_channels=in_channels, dl_workers=get_dl_workers(), encoder_weights=encoder_weights,
                classes = num_classes)

        summary(m, (256, 256, in_channels), device='cpu')

    return m

if __name__ == '__main__':
    seed_everything(seed=45)
    
    # where to store model params
    model_dir = "{}/pretrain-{}".format(model_output_parent, get_time())
    os.makedirs(model_dir, exist_ok=True)
    
    dataset = get_dataset(dataset_dir, model_dir)
    batch_size = get_batch_size()

    # create model
    #model = UNet(datasets, backbone=backbone, encoder_weights=encoder_weights, batch_size=batch_size, lr=lr, classes=2)
    model = get_model(dataset, batch_size)

    # save params
    note = input("Enter title for this training run:")
    params = "note: {}\ndataset: {}\nbatch_size: {}\nbackbone: {}\nWL: {}\nWW: {}\nmean: {}\nstd: {}\ntile_size: 64\nLR: {}\n".format(
        note, dataset_dir, batch_size, backbone, WL, WW, mean, std, lr
    )

    params += "\nin_channels: {}\npre_train: {}\nencoder_weights: {}\nloss: {}\noptimizer params: {}".format(
                    in_channels, pre_train, encoder_weights, loss, optimizer_params)

    params += "\nn_shuffles: {} (for jigsaw pretraining only - n shuffles per epoch)\nnum_permutations: {}".format(num_shuffles, n_perms)

    with open("{}/{}".format(model_dir, params_file), "w") as f:
        f.write(params)

    train_model(model, model_dir)
