import albumentations as A
from albumentations.augmentations.transforms import CenterCrop
from albumentations.core.composition import set_always_apply
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import sys
sys.path.append('./')

from data.CTDataSet import CTDicomSlices, CTDicomSlicesJigsaw, CTDicomSlicesFelzenszwalb
from data.CustomTransforms import Window, Imagify, Normalize
from models.ResNet_jigsaw import ResnetJigsaw

from constants import Constants

def prompt_for_quit():
    print("Type 'q' to quit.")
    inp = input()

    if inp == 'q':
        sys.exit(0)

def show_dataset(ctds):
    '''
    Shows the jigsaw dataset without collate fn
    For "deep" debugging
    '''
    dl = DataLoader(ctds, batch_size=1, num_workers=0, shuffle=True)

    for slices, mask, img_path, slice_n in dl:
        slices = slices[0] # get rid of batch number
                           # don't squeeze because we need 1 for channel
        mask = mask.permute(1, 2, 0) * 255
        show_images(slices, mask, img_path, slice_n)
        prompt_for_quit()

def show_images(slices, mask, img_path, slice_n):
    fig = plt.figure(figsize=(15, 15))

    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(slices, cmap='gray')

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(slices, cmap='gray')
    ax2.imshow(mask, cmap='jet', alpha=0.5)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(mask, cmap='gray')

    fig.show()

if __name__ == '__main__':
    dataset = '/mnt/g/thesis/ct_only_cleaned_resized_mini/head-neck-radiomics'
    #dataset = Constants.organized_dataset_2
    dcm_list = CTDicomSlices.generate_file_list(dataset)
        #dicom_glob='/*/*/*.dcm')

    prep = transforms.Compose([Window(50, 200), Imagify(50, 200)])
    tsfm = A.Compose([A.SmallestMaxSize(max_size=256, always_apply=True, p=1),
                      A.CenterCrop(256, 256, always_apply=True, p=1.0)])

    #ctds = CTDicomSlices(dcm_list, preprocessing=prep, resize_transform=tsfm,
    #                n_surrounding=0)# , felz_crop=True)
    
    ctds = CTDicomSlices(dcm_list, n_surrounding=0, mask_is_255=False)

    show_dataset(ctds)