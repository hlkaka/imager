import sys
import matplotlib.pyplot as plt

sys.path.append('data/')

from Felz_crop_and_masks import create_dataset
from CTDataSet import CTDicomSlicesFelzSaving
from DatasetPreprocessor import DatasetPreprocessor

'''
This script visualizes masks created with Felzenszwalb segmentation and super pixels after
Felz cropping.
'''

def plot_slices_and_mask(image, mask, super_pixels = None):
    mask = mask * 51

    fig = plt.figure(figsize=(15,15))
    cols, rows = 2, 1

    fig.add_subplot(rows, cols, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Image")
    if super_pixels is not None:
        for s in super_pixels:
            plt.plot(s[1], s[0], 'or')
            print("({}, {})".format(s[0], s[1]))

    fig.add_subplot(rows, cols, 2)
    plt.imshow(mask, cmap='gray')
    if super_pixels is not None:
        for s in super_pixels:
            plt.plot(s[1], s[0], 'or')

    plt.title("Mask")
    
    plt.show()

if __name__ == '__main__':
    ds = create_dataset()
    dp = DatasetPreprocessor(ds, '/mnt/g/thesis/ct_only_cleaned', num_workers = 1,
                             shuffle = True)

    while True:
        image, mask, super_pixels = dp.process_next_image()

        if image is None or mask is None:
            continue

        plot_slices_and_mask(image.cpu().numpy(), mask.cpu().numpy(), super_pixels)

        print("Type 'q' to quit.")
        inp = input()

        if inp == 'q':
            break