import albumentations as A
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

    img0 = slices[:,:,0]
    img1 = slices[:,:,1]
    img2 = slices[:,:,2]

    img0 = np.repeat(img0[..., np.newaxis], 3, axis=2)
    img1 = np.repeat(img1[..., np.newaxis], 3, axis=2)
    img2 = np.repeat(img2[..., np.newaxis], 3, axis=2)

    ax = fig.add_subplot(2, 3, 1)
    ax.imshow(img0, cmap='gray')

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(img1, cmap='gray')

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(img2, cmap='gray')

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(slices, cmap='gray')
    ax4.imshow(mask, cmap='jet', alpha=0.5)

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(mask, cmap='gray')

    fig.show()

if __name__ == '__main__':
    dataset = '/home/hussam/imager/organized_dataset_2'
    #dataset = Constants.organized_dataset_2
    dcm_list = CTDicomSlices.generate_file_list(dataset)
        #dicom_glob='/*/*/*.dcm')

    prep = transforms.Compose([Window(50, 200), Imagify(50, 200)])
    tsfm = A.Compose([A.SmallestMaxSize(max_size=256, always_apply=True, p=1),
                      A.CenterCrop(256, 256, always_apply=True, p=1.0)])

    #ctds = CTDicomSlices(dcm_list, preprocessing=prep, resize_transform=tsfm,
    #                n_surrounding=0)# , felz_crop=True)
    
    ctds = CTDicomSlices(dcm_list, n_surrounding=1, mask_is_255=False, same_img_all_channels = True)

    show_dataset(ctds)