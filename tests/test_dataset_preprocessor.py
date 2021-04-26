import sys
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

sys.path.append('data/')

from CTDataSet import CTDicomSlices
from CustomTransforms import Window
from CustomTransforms import Imagify
from CustomTransforms import TorchFunctionalTransforms as TFT
from DatasetPreprocessor import DatasetPreprocessor
import albumentations as A

def create_dataset(dcm_list):
    prep = transforms.Compose([Window(50, 250), Imagify(50, 250)])
    
    resize_transform = A.Compose([A.Resize(512, 512)])

    _msk_trfm = [A.HorizontalFlip()]
    img_trfm = A.Compose(_msk_trfm + [A.GaussNoise()])

    ctds = CTDicomSlices(dcm_list, shuffle=True, classic_segments=True,
            resize_transform=resize_transform, preprocessing=prep,
            transform = img_trfm, n_surrounding=0, trim_edges=False)

    return ctds

def plot_slices_and_mask(image, mask):
    mask = mask * 63

    fig = plt.figure(figsize=(15,15))
    cols, rows = 2, 1

    fig.add_subplot(rows, cols, 1)
    plt.imshow(image)
    plt.title("Image")

    
    fig.add_subplot(rows, cols, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Mask")
    
    plt.show()

if __name__ == '__main__':
    only_positive = True
    dataset = 'E:/thesis/ct_only_filtered_2'
    dcm_list = CTDicomSlices.generate_file_list(dataset, dicom_glob='*/*/*/*.dcm')

    ds = create_dataset(dcm_list)

    dp = DatasetPreprocessor(ds, 'E:/thesis/ct_only_cleaned', 512, 512)

    run_all = True

    if run_all:
        dp.process_dataset()
    else:
        while True:
            image, mask = dp.process_next_image()

            if image is None or mask is None:
                continue

            plot_slices_and_mask(image.cpu().numpy(), mask.cpu().numpy())

            print("Type 'q' to quit.")
            inp = input()

            if inp == 'q':
                break