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

sys.path.append('data/')
from CTDataSet import CTDicomSlices, DatasetManager
from CustomTransforms import Window
from CustomTransforms import Imagify

sys.path.append('models/')
from UNet_L import UNet

# Assumes holdout was done
# Train/val/test split
# Create dataset
# Create dataloader
# Create model
# Run model

dataset = "../organized_dataset"
model_output_parent = "../model_runs"

new_ds_split = True

val_frac = 0.112
test_frac = 0.112

train_list = "train.txt"
val_list = "val.txt"
test_list = "test.txt"
params_file = "params.txt" # where to save params for this run

batch_size = 2

backbone = 'resnet34'
encoder_weights = 'imagenet'

WL = 50
WW = 200

img_size = 256

lr = 0.0001

n_epochs = 10

def to_float(x, **kwargs):
    return x.astype('float32')

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

    preprocess_fn = get_preprocessing_fn(backbone, pretrained=encoder_weights)

    _image_transforms = [
        Window(WL, WW),
        Imagify(WL, WW),
        preprocess_fn,
        to_float,
    ]

    _img_mask_tsfm = A.Compose([A.Resize(img_size, img_size)],
            additional_targets={"image1": 'image', "mask1": 'mask'})

    image_transforms = transforms.Compose(_image_transforms)
    img_mask_tsfm = A.Compose(_img_mask_tsfm)

    # create ds
    train_dicoms, val_dicoms, test_dicoms = dsm.get_dicoms()
    datasets = {}
    datasets['train'] = CTDicomSlices(train_dicoms, preprocessing = image_transforms, img_and_mask_transform = img_mask_tsfm)
    datasets['val'] = CTDicomSlices(val_dicoms, preprocessing = image_transforms, img_and_mask_transform = img_mask_tsfm)
    datasets['test'] = CTDicomSlices(test_dicoms, preprocessing = image_transforms, img_and_mask_transform = img_mask_tsfm)

    # create model
    model = UNet(datasets, batch_size=batch_size, lr=lr)

    # save params
    params = "batch_size: {}\nbackbone: {}\nencoder_weights: {}\nWL: {}\nWW: {}\nimg_size: {}\nLR: {}".format(
        batch_size, backbone, encoder_weights, WL, WW, img_size, lr
    )

    with open("{}/{}".format(model_dir, params_file), "w") as f:
        f.write(params)

    # run
    trainer = Trainer(gpus=0, default_root_dir=model_dir, max_epochs=n_epochs)
    trainer.fit(model)
    trainer.test()