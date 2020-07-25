'''
Script to run the model
'''
import os
import sys
from pytorch_lightning import Trainer
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

# Assumes holdout was done
# Train/val/test split
# Create dataset
# Create dataloader
# Create model
# Run model

dataset = "../organized_dataset_2"
model_output_parent = "../model_runs"

new_ds_split = True

val_frac = 0.112
test_frac = 0.112

train_list = "train.txt"
val_list = "val.txt"
test_list = "test.txt"
params_file = "params.txt" # where to save params for this run

backbone = 'resnet34'
encoder_weights = 'imagenet'

WL = 50
WW = 200

img_size = 256

lr = 0.0001

cpu_batch_size = 2
gpu_batch_size = 64

n_epochs = 10

# Augmentations
rotate=30
translate=(0.2, 0.2)
scale=(0.8, 1.3)
shear=(7, 7)
gaussian_noise_std = 0.3

def get_time():
    now = datetime.now()
    return now.strftime("%Y-%m-%d-%H:%M:%S")

if __name__ == '__main__':
    # Manage patient splits
    if new_ds_split:
        dsm = DatasetManager.generate_train_val_test(dataset, val_frac, test_frac)
    else:
        dsm = DatasetManager.load_train_val_test(dataset, train_list, val_list, test_list)

    # where to store model params
    model_dir = "{}/{}".format(model_output_parent, get_time())
    os.makedirs(model_dir, exist_ok=True)
    dsm.save_lists(model_dir)

    # Setup trainer
    if torch.cuda.is_available():
        batch_size = gpu_batch_size    
        trainer = Trainer(gpus=1, default_root_dir=model_dir, max_epochs=n_epochs)
    else:
        batch_size = cpu_batch_size
        trainer = Trainer(gpus=0, default_root_dir=model_dir, max_epochs=n_epochs)

    #preprocess_fn = get_preprocessing_fn(backbone, pretrained=encoder_weights)

    img_mask_tsfm = A.Compose([A.Resize(img_size, img_size)],
            additional_targets={"image1": 'image', "mask1": 'mask'})

    # create ds
    train_dicoms, val_dicoms, test_dicoms = dsm.get_dicoms()
    datasets = {}
    datasets['train'] = CTDicomSlices(train_dicoms, img_and_mask_transform = img_mask_tsfm)
    datasets['val'] = CTDicomSlices(val_dicoms, img_and_mask_transform = img_mask_tsfm)
    datasets['test'] = CTDicomSlices(test_dicoms, img_and_mask_transform = img_mask_tsfm)

    optimizer_params = {
            'factor': 0.5,
            'patience': 5, 
            'cooldown': 5, 
            'min_lr': 1e-6
    }

    # create model
    #model = UNet(datasets, backbone=backbone, encoder_weights=encoder_weights, batch_size=batch_size, lr=lr, classes=2)
    model = UNet_m(datasets, lr=lr, batch_size = batch_size, gaussian_noise_std = gaussian_noise_std,
                 degrees=rotate, translate=translate, scale=scale, shear=shear, optimizer_params=optimizer_params)

    # save params
    params = "dataset: {}\nbatch_size: {}\nbackbone: {}\nencoder_weights: {}\nWL: {}\nWW: {}\nimg_size: {}\nLR: {}\n".format(
        dataset, batch_size, backbone, encoder_weights, WL, WW, img_size, lr
    )
    params += "\n\n# AUGMENTATIONS\n\n rotation degrees: {}\ntranslate: {}\nscale: {}\nshear: {}\nGaussian noise: {}".format(
        rotate, translate, scale, shear, gaussian_noise_std
    )

    params += "\nOptimizer params: {}".format(optimizer_params)
    
    with open("{}/{}".format(model_dir, params_file), "w") as f:
        f.write(params)

    # Run
    trainer.fit(model)
    trainer.test()