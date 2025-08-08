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

from skimage.segmentation import felzenszwalb, chan_vese

import sys
sys.path.append('data/')

from CTDataSet import CTDicomSlices, CTDicomSlicesFelzenszwalb
from CustomTransforms import Window
from CustomTransforms import Imagify
from CustomTransforms import TorchFunctionalTransforms as TFT

import timeit

# TODO:
# 1. Separate felzenswalb
# 2. Strip image by calling smart strip
# 3. Pick 4 corner super pixels
# 4. Do in CTDataSet and generate dataset


def plot_slices_and_mask(slices, mask, text = "", segments = None, f_mask = None):
    mask = mask * 255

    extra_row = 0 if segments is None else 1

    fig = plt.figure(figsize=(15,15))
    cols, rows = 2, slices.shape[0] // 2 + extra_row + 1  # + 1 for mask

    for i in range(slices.shape[0]):
        cur_slice = np.expand_dims(slices[i, :, :], 0)
        cur_slice = np.concatenate((cur_slice, cur_slice, cur_slice), axis=0)

        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(np.moveaxis(cur_slice, 0, -1))
        plt.title(text)

    
    fig.add_subplot(rows, cols, slices.shape[0] + 1)
    plt.imshow(mask[0], cmap='gray')
    plt.title("Label mask")

    if extra_row == 1:
        fig.add_subplot(rows, cols, slices.shape[0] + 2)
        plt.imshow(segments)
        plt.title("All segments")

        fig.add_subplot(rows, cols, slices.shape[0] + 3)
        plt.imshow(f_mask, cmap='gray')
        plt.title("Self-supervised mask")
    
    plt.show()

def cpu_transforms(dcm_list) -> DataLoader:
    prep = transforms.Compose([Window(50, 250), Imagify(50, 250)])
    
    _img_trfm = [A.GaussNoise()]
                 #A.RandomBrightnessContrast()]

    #img_trfm = A.Compose(_img_trfm)
    

    resize_transform = A.Compose([A.Resize(256, 256)])

    _msk_trfm = [#A.Resize(256, 256),
                 #A.ElasticTransform(alpha_affine=10, p=0.8),
                 A.HorizontalFlip(),
                 #A.OpticalDistortion(),
                 #A.Rotate(limit=30, p=1)
                 ]

    img_trfm = A.Compose(_msk_trfm + [A.GaussNoise()])

    msk_trfm = A.Compose(_msk_trfm,
            additional_targets={"image1": 'image', "mask1": 'mask'})
    #prep = get_preprocessing_fn('resnet34', pretrained='imagenet')

    ctds = CTDicomSlicesFelzenszwalb(dcm_list, resize_transform=resize_transform, preprocessing=prep, transform = img_trfm, n_surrounding=0, trim_edges=False)

    dl = DataLoader(ctds, batch_size=1, num_workers = 0, shuffle=True)

    return dl

def do_inplace_transforms(images, masks):
    ra = kornia.augmentation.RandomAffine(degrees=30, scale=(0.8, 1.3), shear=(7, 7))
    rf = kornia.augmentation.RandomHorizontalFlip()
    with torch.no_grad():
        TFT.Window(images, 50, 200)
        TFT.Imagify(images, 50, 200)
        TFT.GaussianNoise(images, std = 3)
        
        #params = ra.generate_parameters(images.shape)
        #images = ra(images, params)
        #masks = ra(masks, params)

        params = rf.generate_parameters(images.shape)
        images = rf(images, params)
        masks = rf(masks, params)
    return images, masks

def gpu_transforms(dcm_list) -> DataLoader:
    msk_trfm = A.Compose([A.Resize(256, 256)],
            additional_targets={"image1": 'image', "mask1": 'mask'})

    ctds = CTDicomSlices(dcm_list, shuffle=True, img_and_mask_transform = msk_trfm, n_surrounding=0, trim_edges=False, self_supervised_mask=True)#, preprocessing = prep)
    
    dl = DataLoader(ctds, batch_size=1, num_workers = 0, shuffle=True)

    return dl

def get_felzenszwalb(slices :np.array) -> np.array :
    slices = slices.cpu().detach().numpy()

    mid_slice = slices.shape[0] // 2

    segments = felzenszwalb(slices[mid_slice,:,:], scale=150, sigma=0.7, min_size=50)
    #segments = chan_vese(slices[mid_slice,:,:], mu=0.1)

    selected_pixels = np.array([[5/16, 5/16], [5/16, 11/16], [11/16, 5/16], [11/16, 11/16]]) @ np.array([[256, 0], [0, 256]])    # don't hard code image resolution
    selected_pixels = selected_pixels.astype('int32')

    selected_segments = [segments[tuple(sp)] for sp in selected_pixels]

    pre_mask = [segments == ss for ss in selected_segments]

    mask = np.logical_or.reduce(pre_mask)

    bg_segment = int(np.floor(np.mean(segments[0])))
    mid_col = segments.shape[1] // 2

    start = timeit.default_timer()

    max_gap = smart_strip_edges(segments, bg_segment, 1)
    max_h_gap = smart_strip_edges(segments, bg_segment, 0)

    stop = timeit.default_timer()

    print("Step: {} - Runtime: {}".format(3, stop-start))

    return segments[max_gap[0]:max_gap[1], max_h_gap[0]:max_h_gap[1]], mask[max_gap[0]:max_gap[1], max_h_gap[0]:max_h_gap[1]]

def smart_strip_edges(segments, bg_segment, dim, step = 3, buffer = 10):
    '''
    Uses the background segment of felzenswalb to strip the vertical and horizontal edges of images.
    segments: felzenswalb segmentation
    bg_segment: usually 0. The segment label assigned to background.
    dim: dimension to act on. 1 strips top and bottom, 0 strips left and right.
    step: how many pixels to scan every step. Default is 3
    buffer: how many pixels to leave before stripping
    '''
    mid_line = segments.shape[dim] // 2

    if dim == 0:
        mid_line_pixels = segments[mid_line]
    elif dim == 1:
        mid_line_pixels = segments[:,mid_line]

    gaps = []

    for i in range(0, segments.shape[dim], step):
        p = mid_line_pixels[i]
        if p != bg_segment:
            if len(gaps) > 0 and gaps[-1][1] == i:
                gaps[-1][1] = i + step
            else:
                gaps.append([i, i+step])
    
    gap_lengths = [g[1] - g[0] for g in gaps]
    index_max_gap = gap_lengths.index(max(gap_lengths))
    max_gap = gaps[index_max_gap]

    max_gap = [max_gap[0] - buffer, max_gap[1] + buffer]

    return max_gap

use_gpu_transforms = False

if __name__ == '__main__':
    only_positive = True
    dataset = '/mnt/d/thesis/ct_only_filtered_2/head-neck-pet-ct'
    dcm_list = CTDicomSlices.generate_file_list(dataset, dicom_glob='*/*/*.dcm')

    if use_gpu_transforms:
        ct_dl = gpu_transforms(dcm_list)
    else:
        ct_dl = cpu_transforms(dcm_list)

    for slices, mask, img_path, slice_n in ct_dl:
        slices = slices.permute(0, 3, 1, 2)
        if use_gpu_transforms:
            slices, mask = do_inplace_transforms(slices, mask)
            mask = mask[0]

        slices = slices[0]
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

        segments, f_mask = get_felzenszwalb(slices)

        plot_slices_and_mask(slices.cpu().numpy(), mask.cpu().numpy(), "Pt: {} - SN: {}".format(pt_id, slice_n), segments=segments, f_mask=f_mask)


        print("Type 'q' to quit.")
        inp = input()

        if inp == 'q':
            break
