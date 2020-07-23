import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import albumentations as A
from torchvision import transforms
from segmentation_models_pytorch.encoders import get_preprocessing_fn

import sys
sys.path.append('data/')

from CTDataSet import CTDicomSlices
from CustomTransforms import Window
from CustomTransforms import Imagify

def plot_slices_and_mask(slices, mask, text = ""):
    mask = mask * 255

    fig = plt.figure(figsize=(15,15))
    cols, rows = 2, 2

    for i in range(slices.shape[2]):
        cur_slice = np.expand_dims(slices[:, :, i], 2)
        cur_slice = np.concatenate((cur_slice, cur_slice, cur_slice), axis=2)

        fig.add_subplot(2, 2, i + 1)
        plt.imshow(cur_slice)

    fig.add_subplot(2, 2, slices.shape[2] + 1)
    plt.imshow(mask)
    plt.title(text)
    plt.show()

if __name__ == '__main__':
    only_positive = True
    dataset = '/home/hussam/organized_dataset/'
    dcm_list = CTDicomSlices.generate_file_list(dataset)

    trfm = transforms.Compose([Window(50, 200), Imagify(50, 200)])
    msk_trfm = A.Compose([A.Resize(256, 256)],
            additional_targets={"image1": 'image', "mask1": 'mask'})
    prep = get_preprocessing_fn('resnet34', pretrained='imagenet')

    ctds = CTDicomSlices(dcm_list, shuffle=True, transform=trfm, img_and_mask_transform = msk_trfm, preprocessing = prep)

    for slices, mask, img_path, slice_n in ctds:
        pt_id = os.path.dirname(os.path.dirname(img_path))
        pt_id = os.path.basename(pt_id)

        if only_positive:
            if np.sum(mask) == 0:
                continue

        slices = slices - np.min(slices)
        slices = slices * 255.0 / np.max(slices)
        slices = slices.astype('uint8')

        plot_slices_and_mask(slices, mask, "Pt: {} - SN: {}".format(pt_id, slice_n))

        print("Type 'q' to quit.")
        inp = input()

        if inp == 'q':
            break
