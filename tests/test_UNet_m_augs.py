import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from torch.utils.data import DataLoader
import torch

import sys
sys.path.append('models')
from run_model import *

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
    plt.imshow(mask)
    plt.title(text)
    plt.show()

if __name__ == '__main__':
    only_positive = True
    
    datasets = get_datasets(model_dir = None)
    batch_size = 1

    model = get_model(datasets, batch_size)
    ct_dl = model.train_dataloader()

    for slices, mask, img_path, slice_n in ct_dl:
        if only_positive:
            if torch.sum(mask) == 0:
                continue

        # Preprocessing any image
        slices, mask = model.preprocessing(slices, mask, b_h_w_c = True)
        
        # Preprocessing for training images only
        slices, mask = model.do_train_augmentations(slices, mask)

        slices = slices[0]
        mask = mask[0]
        img_path = img_path[0]
        slice_n = slice_n[0]

        pt_id = os.path.dirname(os.path.dirname(img_path))
        pt_id = os.path.basename(pt_id)

        slices = slices - torch.min(slices)
        slices = slices * 255.0 / torch.max(slices)
        slices = slices.type(torch.uint8)

        plot_slices_and_mask(slices.cpu().numpy(), mask.cpu().numpy(), "Pt: {} - SN: {}".format(pt_id, slice_n))

        print("Type 'q' to quit.")
        inp = input()

        if inp == 'q':
            break
