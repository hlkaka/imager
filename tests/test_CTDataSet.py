import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import albumentations as A
from torchvision import transforms
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from torch.utils.data import DataLoader
import torch
import kornia

import sys
sys.path.append('data/')

from CTDataSet import CTDicomSlices
from CustomTransforms import Window
from CustomTransforms import Imagify
from CustomTransforms import TorchFunctionalTransforms as TFT

def plot_slices_and_mask(slices, mask, text = ""):
    mask = mask * 255

    fig = plt.figure(figsize=(15,15))
    cols, rows = 2, 2

    for i in range(slices.shape[0]):
        cur_slice = np.expand_dims(slices[i, :, :], 0)
        cur_slice = np.concatenate((cur_slice, cur_slice, cur_slice), axis=0)

        fig.add_subplot(2, 2, i + 1)
        plt.imshow(np.moveaxis(cur_slice, 0, -1))

    fig.add_subplot(2, 2, slices.shape[0] + 1)
    plt.imshow(mask[0])
    plt.title(text)
    plt.show()

def cpu_transforms(dcm_list) -> DataLoader:
    prep = transforms.Compose([Window(50, 200), Imagify(50, 200)])
    
    _img_trfm = [A.GaussNoise()]
                 #A.RandomBrightnessContrast()]

    img_trfm = A.Compose(_img_trfm)

    _msk_trfm = [A.Resize(256, 256),
                 A.ElasticTransform(alpha_affine=10, p=0.8),
                 A.HorizontalFlip(),
                 A.OpticalDistortion(),
                 A.Rotate(limit=30, p=1)]

    msk_trfm = A.Compose(_msk_trfm,
            additional_targets={"image1": 'image', "mask1": 'mask'})
    #prep = get_preprocessing_fn('resnet34', pretrained='imagenet')

    ctds = CTDicomSlices(dcm_list, shuffle=True, preprocessing=prep, transform = img_trfm, img_and_mask_transform = msk_trfm)#, preprocessing = prep)

    dl = DataLoader(ctds, batch_size=1, num_workers = 0, shuffle=True)

    return dl

def do_inplace_transforms(images, masks):
    ra = kornia.augmentation.RandomAffine(degrees=30, translate=(0.2, 0.2), scale=(0.8, 1.3), shear=(7, 7))
    rf = kornia.augmentation.RandomHorizontalFlip()
    with torch.no_grad():
        TFT.Window(images, 50, 200)
        TFT.Imagify(images, 50, 200)
        TFT.GaussianNoise(images, std = 3)
        
        params = ra.generate_parameters(images.shape)
        images = ra(images, params)
        masks = ra(masks, params)

        params = rf.generate_parameters(images.shape)
        images = rf(images, params)
        masks = rf(masks, params)
    return images, masks

def gpu_transforms(dcm_list) -> DataLoader:
    msk_trfm = A.Compose([A.Resize(256, 256)],
            additional_targets={"image1": 'image', "mask1": 'mask'})

    ctds = CTDicomSlices(dcm_list, shuffle=True, img_and_mask_transform = msk_trfm)#, preprocessing = prep)
    
    dl = DataLoader(ctds, batch_size=1, num_workers = 0, shuffle=True)

    return dl

use_gpu_transforms = True

if __name__ == '__main__':
    only_positive = True
    dataset = '/home/hussam/organized_dataset_2/'
    dcm_list = CTDicomSlices.generate_file_list(dataset)

    if use_gpu_transforms:
        ct_dl = gpu_transforms(dcm_list)
    else:
        ct_dl = cpu_transforms(dcm_list)

    for slices, mask, img_path, slice_n in ct_dl:
        slices = slices.permute(0, 3, 1, 2)
        if use_gpu_transforms:
            
            slices, mask = do_inplace_transforms(slices, mask)
            #pass

        slices = slices[0]
        mask = mask[0]
        img_path = img_path[0]
        slice_n = slice_n[0]

        pt_id = os.path.dirname(os.path.dirname(img_path))
        pt_id = os.path.basename(pt_id)

        if only_positive:
            if torch.sum(mask) == 0:
                continue

        slices = slices - torch.min(slices)
        slices = slices * 255.0 / torch.max(slices)
        slices = slices.type(torch.uint8)

        plot_slices_and_mask(slices.cpu().numpy(), mask.cpu().numpy(), "Pt: {} - SN: {}".format(pt_id, slice_n))

        print("Type 'q' to quit.")
        inp = input()

        if inp == 'q':
            break
