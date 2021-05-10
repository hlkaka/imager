import albumentations as A
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob

import sys
sys.path.append('data/')

from CTDataSet import CTDicomSlices, CTDicomSlicesJigsaw
from CustomTransforms import Window
from CustomTransforms import Imagify

def prompt_for_quit():
    print("Type 'q' to quit.")
    inp = input()

    if inp == 'q':
        sys.exit(0)

def show_jigsaw_dataset(dcm_list):
    prep = transforms.Compose([Window(50, 200), Imagify(50, 200)])
    ctds = CTDicomSlicesJigsaw(dcm_list, preprocessing=prep, trim_edges=True,
            return_tile_coords=True, normalize_tiles=False, max_pixel_value=255)
    dl = DataLoader(ctds, batch_size=1, num_workers=0, shuffle=True)

    for image, img_path, slice_n, tiles, coords in dl:
        fig = plt.figure(figsize=(15, 15))

        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(image.squeeze(), cmap='gray')
        for r in coords:
            # reversing list because matplot lib is y-x
            # also, x-axis is from bottom
            r = r.squeeze().detach().numpy().tolist()
            r.reverse()
            p = patches.Rectangle(r, ctds.tile_size, ctds.tile_size, linewidth=1,
                                edgecolor='g', facecolor='none')
            ax.add_patch(p)

        tiles = tiles.detach().numpy().squeeze()
        puzzle = np.empty((ctds.tile_size * ctds.snjp, ctds.tile_size * ctds.snjp))

        for i in range(ctds.snjp):
            for j in range(ctds.snjp):
                tl = [i * ctds.tile_size, j * ctds.tile_size]
                br = [tl[0] + ctds.tile_size, tl[1] + ctds.tile_size]

                puzzle[tl[0]:br[0], tl[1]:br[1]] = tiles[i * ctds.snjp + j]

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(puzzle, cmap='gray')

        fig.show()
        prompt_for_quit()

if __name__ == '__main__':
    #dataset = '/mnt/g/thesis/ct_only_cleaned'
    dataset = '/mnt/g/thesis/ct_only_filtered_2/head-neck-radiomics'
    dcm_list = CTDicomSlices.generate_file_list(dataset,
        dicom_glob='/*/*.dcm')

    show_jigsaw_dataset(dcm_list)