'''
Script to run the model
'''
import os
import sys

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.plugins import DDPPlugin
from torchvision import transforms
from segmentation_models_pytorch.base.heads import SegmentationHead

import albumentations as A
from datetime import datetime

sys.path.append('.')
from data.CTDataSet import CTDicomSlices, DatasetManager
from data.CustomTransforms import Window, Imagify, Normalize

from models.UNet_L import UNet
from models.UNet_mateuszbuda import UNet_m
from models.ResNet_jigsaw import ResnetJigsaw

from constants import Constants

from torchsummary import summary
from pytorch_lightning import Trainer, loggers as pl_loggers

# Consider removing sys.path.append and trying to refer to scripts as data.CTDataSet etc

# Assumes holdout was done
# Train/val/test split
# Create dataset
# Create dataloader
# Create model
# Run model

dataset = Constants.organized_dataset_2
model_output_parent = Constants.model_outputs

val_frac = 0.112
test_frac = 0.112

params_file = "params.txt" # where to save params for this run

backbone = 'resnet34'
encoder_weights = 'imagenet'

WL = 50
WW = 200

img_size = 256

lr = 0.00005
freeze_backbone = False
freeze_n_layers = 0

cpu_batch_size = 2
gpu_batch_size = 128

n_epochs = 10

in_channels = 3 

# Augmentations
rotate=15
translate=(0.1, 0.1)
scale=(0.9, 1.1)

optimizer_params = {
        'factor': 0.5,
        'patience': 5, 
        'cooldown': 0, 
        'min_lr': 1e-6
}

mean, std = [61.0249], [78.3195] 

resnet_checkpoint = None #Constants.pretrained_jigsaw
unet_checkpoint = Constants.pretrained_unet_imagenet

train_frac = 1.0

def get_time():
    now = datetime.now()
    return now.strftime("%Y-%m-%d-%H:%M:%S")

def get_datasets(model_dir = None, new_ds_split = True,
                    train_list = "train.txt", val_list = "val.txt", test_list = "test.txt"):
    '''
    Builds the necessary datasets
    model_dir is where model parameters will be stored
    new_ds_split creates a new train/val/test split if True, and loads the relevant folders if false
    dslist_in_pt_dir uses the pt dir as a reference for where the patient lists are located
    '''

    # Manage patient splits
    if new_ds_split:
        dsm = DatasetManager.generate_train_val_test(dataset, val_frac, test_frac)
        if model_dir is not None:
            dsm.save_lists(model_dir)
    else:
        dsm = DatasetManager.load_train_val_test(dataset, train_list, val_list, test_list)
    
    #preprocess_fn = get_preprocessing_fn(backbone, pretrained=encoder_weights)

    prep = transforms.Compose([Window(WL, WW), Imagify(WL, WW), Normalize(mean, std)])

    resize_tsfm = A.Compose([A.Resize(img_size, img_size)],
            additional_targets={"image1": 'image', "mask1": 'mask'})

    img_mask_tsfm = A.Compose([
                    A.ShiftScaleRotate(shift_limit=translate, scale_limit=scale, rotate_limit=rotate),
                    A.HorizontalFlip()],
            additional_targets={"image1": 'image', "mask1": 'mask'})

    # create ds
    train_dicoms, val_dicoms, test_dicoms = dsm.get_dicoms(train_frac = train_frac)
    
    datasets = {}
    datasets['train'] = CTDicomSlices(train_dicoms, preprocessing = prep,
                        resize_transform = resize_tsfm, img_and_mask_transform = img_mask_tsfm, n_surrounding=in_channels // 2)
    datasets['val'] = CTDicomSlices(val_dicoms, preprocessing = prep, resize_transform = resize_tsfm, n_surrounding=in_channels // 2)
    datasets['test'] = CTDicomSlices(test_dicoms, preprocessing = prep, resize_transform = resize_tsfm, n_surrounding=in_channels // 2)

    return datasets

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
        #trainer = Trainer(gpus=Constants.n_gpus, distributed_backend='ddp', logger = tb_logger, precision=16, default_root_dir=model_dir, max_epochs=n_epochs)
        trainer = Trainer(gpus=Constants.n_gpus, plugins=DDPPlugin(find_unused_parameters=False), accelerator='ddp_spawn', precision=16, logger = tb_logger, default_root_dir=model_dir, max_epochs=n_epochs)
    else:
        trainer = Trainer(gpus=0, default_root_dir=model_dir, logger = tb_logger, distributed_backend='ddp_spawn', max_epochs=n_epochs)
    
    trainer.fit(model)
    trainer.test()

def get_dl_workers():
    dl_workers = input("Number of DataLoader workers (usually = CPU cores). No error checking:")

    try:
        dl_workers = int(dl_workers)
    except:
        dl_workers = 1

    print("Using {} DL workers".format(dl_workers))

    return dl_workers

def get_model(datasets, batch_size):
    # UNet Mateuszbuda
    #return UNet_m(datasets, lr=lr, batch_size = batch_size, gaussian_noise_std = gaussian_noise_std,
    #             degrees=rotate, translate=translate, scale=scale, shear=shear, optimizer_params=optimizer_params)
    
    # UNet from segmentation models package
    #training_mean, training_std = datasets['train'].calculate_ds_mean_std()

    if resnet_checkpoint is not None:
        m = UNet(datasets, backbone=backbone, batch_size=batch_size, optimizer_params=optimizer_params,
                in_channels=in_channels, dl_workers=get_dl_workers(), encoder_weights=encoder_weights, lr=lr)

        pretrained = ResnetJigsaw.load_from_checkpoint(resnet_checkpoint, datasets= datasets['train'], map_location='cpu')

        # This commented line is for viewing past models
        #pretrained_2 = UNet.load_from_checkpoint('/mnt/e/HNSCC dataset/trained_models/14 - 100_epochs_resnet34_encoder_nonfrozen_single_slice/lightning_logs/version_0/checkpoints/epoch=72.ckpt', strict=False, datasets= datasets['train'], map_location='cpu', in_channels=1)
        
        m.smp_unet.encoder.conv1 = pretrained.resnet.conv1
        m.smp_unet.encoder.bn1 = pretrained.resnet.bn1
        m.smp_unet.encoder.relu = pretrained.resnet.relu
        m.smp_unet.encoder.maxpool = pretrained.resnet.maxpool
        m.smp_unet.encoder.layer1 = pretrained.resnet.layer1
        m.smp_unet.encoder.layer2 = pretrained.resnet.layer2
        m.smp_unet.encoder.layer3 = pretrained.resnet.layer3
        m.smp_unet.encoder.layer4 = pretrained.resnet.layer4
    
    elif unet_checkpoint is not None:
        pretrained = UNet.load_from_checkpoint(unet_checkpoint, datasets=datasets, map_location='cpu', dl_workers=get_dl_workers(), batch_size=batch_size,
                                backbone=backbone, in_channels=in_channels, encoder_weights=None, optimizer_params=optimizer_params, classes=6, lr=lr) # encoder weights are None because they will be loaded. No need to duplicate

        # https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/unet/model.py
        decoder_channels = (256, 128, 64, 32, 16)    # seems to be hardcoded in the UNet constructor
        pretrained.smp_unet.segmentation_head = SegmentationHead(
            in_channels = decoder_channels[-1],
            out_channels = 2,
            activation = 'softmax',   # magic value
            kernel_size = 3,
        )

        m = pretrained
    
    else:
        m = UNet(datasets, backbone=backbone, batch_size=batch_size, optimizer_params=optimizer_params,
                in_channels=in_channels, dl_workers=get_dl_workers(), encoder_weights=encoder_weights, lr=lr)

    if freeze_backbone:       
        # Freeze entire backbone
        for param in m.smp_unet.encoder.parameters():
            param.requires_grad = False

    # Freeze only some layers
    if freeze_n_layers > 0 and not freeze_backbone:
        ct = 0
        for child in m.smp_unet.encoder.children():
            ct += 1
            if ct < freeze_n_layers: # Number of layers to freeze
                for param in child.parameters():
                    param.requires_grad = False

    summary(m, (img_size, img_size, in_channels), device='cpu')

    return m

if __name__ == '__main__':
    seed_everything(seed=45)
    
    # where to store model params
    model_dir = "{}/{}".format(model_output_parent, get_time())
    os.makedirs(model_dir, exist_ok=True)
    
    datasets = get_datasets(model_dir)
    batch_size = get_batch_size()

    # create model
    #model = UNet(datasets, backbone=backbone, encoder_weights=encoder_weights, batch_size=batch_size, lr=lr, classes=2)
    model = get_model(datasets, batch_size)

    # save params
    note = input("Enter title for this training run:")
    params = "note: {}\ndataset: {}\nbatch_size: {}\nbackbone: {}\nencoder_weights: {}\nWL: {}\nWW: {}\nimg_size: {}\nLR: {}\n".format(
        note, dataset, batch_size, backbone, encoder_weights, WL, WW, img_size, lr
    )
    params += "\n\n# AUGMENTATIONS\n\n rotation degrees: {}\ntranslate: {}\nscale: {}\ntrain set fraction: {}".format(
        rotate, translate, scale, train_frac
    )

    params += "\ndataset: {}\nin_channels: {}\nOptimizer params: {}\nFrozen backbone {} OR frozen layers {} ".format(
                    dataset, in_channels, optimizer_params, freeze_backbone, freeze_n_layers)
    
    params += '-- Note if the backbone is frozen, then the number of frozen layers is ignored. Also note frozen layers has a bug that exaggerated the number by 1'

    with open("{}/{}".format(model_dir, params_file), "w") as f:
        f.write(params)

    train_model(model, model_dir)