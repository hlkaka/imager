import albumentations as A
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import sys
sys.path.append('./')

from data.CTDataSet import CTDicomSlices, CTDicomSlicesJigsaw, jigsaw_training_collate
from data.CustomTransforms import Window, Imagify, Normalize
from models.ResNet_jigsaw import ResnetJigsaw

from constants import Constants

def prompt_for_quit():
    print("Type 'q' to quit.")
    inp = input()

    if inp == 'q':
        sys.exit(0)

def show_jigsaw_dataset(ctds):
    '''
    Shows the jigsaw dataset without collate fn
    For "deep" debugging
    '''
    dl = DataLoader(ctds, batch_size=1, num_workers=0, shuffle=True)

    for image, img_path, slice_n, tiles, coords, all_tiles, labels in dl:
        tiles = all_tiles[0, 0]
        print("Label is: {}".format(labels[0,0]))
        show_images(ctds, image, img_path, coords, tiles)
        prompt_for_quit()

def show_jigsaw_training_dataset(ctds, predict=False):
    '''
    Shows the jigsaw dataset without collate fn
    For "deep" debugging
    '''
    dl = DataLoader(ctds, batch_size=1, num_workers=0, shuffle=True, collate_fn=jigsaw_training_collate)

    if predict:
        chkpt = '/mnt/e/HNSCC dataset/trained_models/pretrain_jigsaw_fixed_imagenet/logs/default/version_0/checkpoints/epoch=34-step=85224.ckpt'
        model = ResnetJigsaw.load_from_checkpoint(chkpt, datasets= {'train': ctds}, map_location='cpu', in_channels=3)

    for all_tiles, labels in dl:
        if predict:
            logits = model(all_tiles)
            print("Loss is: {}".format(model.loss(logits, labels)))
            sm = torch.nn.functional.softmax(logits, dim=1)
            prdct_class = torch.argmax(sm)
            print("Prediction is: {} and label is: {}".format(prdct_class, labels))

        show_images(ctds, None, None, None, all_tiles)
        prompt_for_quit()


def show_images(ctds, image, img_path, coords, tiles):
    fig = plt.figure(figsize=(15, 15))

    if image is not None:
        image = ctds.ensure_min_size(image.squeeze().numpy())

        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(image, cmap='gray')
        ax.set_title(img_path)

        for r in coords:
            # reversing list because matplot lib is y-x
            # also, x-axis is from bottom
            r = r.squeeze().detach().numpy().tolist()
            r.reverse()
            p = patches.Rectangle(r, ctds.tile_size, ctds.tile_size, linewidth=1,
                                edgecolor='g', facecolor='none')
            ax.add_patch(p)

    puzzle = ResnetJigsaw.tiles_to_image(tiles, in_channels=1)[0]

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(puzzle[0], cmap='gray')

    fig.show()

if __name__ == '__main__':
    dataset = '/mnt/g/thesis/ct_only_cleaned_mini' #Constants.ct_only_cleaned
    dcm_list = CTDicomSlices.generate_file_list(dataset,
        dicom_glob='/*/*/dicoms/*.dcm')

    prep = transforms.Compose([Window(50, 200), Imagify(50, 200)]) #, Normalize(61.0249, 78.3195)])
    ctds = CTDicomSlicesJigsaw(dcm_list, preprocessing=prep, trim_edges=False,
            return_tile_coords=True, perm_path=Constants.default_perms, n_shuffles_per_image=1)

    #show_jigsaw_dataset(ctds)
    show_jigsaw_training_dataset(ctds, predict=True)