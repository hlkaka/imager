import tqdm
import numpy as np
import torch
import os
import SimpleITK as sitk
from unidecode import unidecode

from torch.utils.data import DataLoader

class DatasetPreprocessor():
    def __init__(self, dataset, output_dir :str, num_workers :int = 1, shuffle = True):
        '''
        this is used only for saving masks
        '''
        self.output_dir = output_dir
        self.dl = DataLoader(dataset, batch_size=1, num_workers=num_workers, shuffle=shuffle)

        self.reset_iterator()

    def reset_iterator(self):
        self.image_iter = iter(self.dl)

    def process_next_image(self):
        ''' Gets the next cropped image and generated mask and saves them '''
        # Be aware that as output passed through Dataloader, metadata
        # values all become lists. So open these lists.
        image, mask, img_path, slice_n, segments, metadata, super_pixels = next(self.image_iter)

        #print(img_path)

        if image is None:
            return None, None, None 

        # Get rid of dim 1 (channel size)
        image = image.squeeze()
        segments = segments.squeeze() 
        mask = mask.squeeze()
        super_pixels = super_pixels.squeeze()

        slice_n = int(slice_n)

        self.save_image_and_mask(image, mask, img_path[0], slice_n, metadata)

        return image, mask, super_pixels

    def process_dataset(self):
        ''' Processes the entire dataset with a tqdm progress bar '''
        n_images = len(self.image_iter)
        with tqdm.tqdm(total=n_images) as progress_bar:
            for _ in range(n_images):
                self.process_next_image()
                progress_bar.update(1)

    def save_image_and_mask(self, image, mask, img_path :str, slice_n :int, metadata :dict):
        ''' Saves the image and mask. For DICOM, metadata is preserved '''
        subdirs = []
        full_study_dir, img_name = os.path.split(img_path)
        ds_dir, study_name = os.path.split(full_study_dir)
        ds_name = os.path.basename(ds_dir)

        dcm_dir = "{}/{}/{}/dicoms".format(self.output_dir, ds_name, study_name)
        mask_dir = "{}/{}/{}/GTV".format(self.output_dir, ds_name, study_name)

        os.makedirs(dcm_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        writer = sitk.ImageFileWriter()

        # DICOMS need the metadata dictionary
        writer.SetFileName("{}/{}.dcm".format(dcm_dir, slice_n))
        output_dcm = sitk.GetImageFromArray(image.type(torch.uint8))
        for k in metadata:
            # Dataloader converts all values to lists. Need to reverse with [0]
            # SITK doesn't support non-ASCII. So remove those characters

            output_dcm.SetMetaData(k, unidecode(metadata[k][0]))
        writer.Execute(output_dcm)

        writer.SetFileName("{}/{}.png".format(mask_dir, slice_n))
        writer.Execute(sitk.GetImageFromArray(mask.type(torch.uint8) * 255))
