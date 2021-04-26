import tqdm
import numpy as np
import torch
import os
import SimpleITK as sitk

from torch.utils.data import DataLoader

class DatasetPreprocessor():
    def __init__(self, dataset, output_dir :str, x_dim :int = 256, y_dim :int = 256,
                 num_workers :int = 1, super_pixels :np.array = None):
        '''
        dirs_to_dcm: specifies how many dirs need to be traversed from dataset directory to get to DICOMs
        for main dataset, this should be 2. for pre-training dataset, this should be 1.
        this is used only for saving masks
        x_dim = size of x dimension in pixels
        y_dim = size of y dimension in pixels
        '''
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.output_dir = output_dir
        self.dl = DataLoader(dataset, batch_size=1, num_workers=num_workers)
        
        if super_pixels is None:
            self.super_pixels = np.array([[5/16, 5/16], [5/16, 11/16], [11/16, 5/16], [11/16, 11/16]])
        else:
            self.super_pixels = super_pixels

        self.reset_iterator()

    def reset_iterator(self):
        self.image_iter = iter(self.dl)

    def process_next_image(self):
        image, segments, img_path, slice_n = next(self.image_iter)

        image = image.squeeze()
        segments = segments.squeeze() # Get rid of dim 1 (channel size)
        slice_n = int(slice_n)

        v_dims = self.get_crops(segments, 1)
        h_dims = self.get_crops(segments, 0)

        if v_dims is None or h_dims is None:
            return None, None

        image = image[v_dims[0]:v_dims[1], h_dims[0]:h_dims[1]]
        segments = segments[v_dims[0]:v_dims[1], h_dims[0]:h_dims[1]]

        mask = self.generate_mask(segments)

        self.save_image_and_mask(image, mask, img_path[0], slice_n)

        return image, mask

    def process_dataset(self):
        n_images = len(self.image_iter)
        with tqdm.tqdm(total=n_images) as progress_bar:
            for _ in range(n_images):
                self.process_next_image()
                progress_bar.update(1)

    def get_crops(self, segments, dim, step = 1, buffer = 20, bg_segment=0):
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
            p = mid_line_pixels.squeeze()[i]
            if p != bg_segment:
                if len(gaps) > 0 and gaps[-1][1] == i:
                    gaps[-1][1] = i + step
                else:
                    gaps.append([i, i+step])
        
        gap_lengths = [g[1] - g[0] for g in gaps]

        if len(gap_lengths) == 0:
            return None

        index_max_gap = gap_lengths.index(max(gap_lengths))
        max_gap = gaps[index_max_gap]

        # Clip it to image dimensions, so we dont have negative indices or indx > dim
        max_gap = [max(0, max_gap[0] - buffer), min(max_gap[1] + buffer, segments.shape[dim])]

        return max_gap

    def save_image_and_mask(self, image, mask, img_path :str, slice_n :int):
        subdirs = []
        full_study_dir, img_name = os.path.split(img_path)
        ds_dir, study_name = os.path.split(full_study_dir)
        ds_name = os.path.basename(ds_dir)

        dcm_dir = "{}/{}/{}/dicoms".format(self.output_dir, ds_name, study_name)
        mask_dir = "{}/{}/{}/GTV".format(self.output_dir, ds_name, study_name)

        os.makedirs(dcm_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        writer = sitk.ImageFileWriter()
        # -4 to drop .dcm and add .png
        writer.SetFileName("{}/{}.png".format(dcm_dir, slice_n))
        writer.Execute(sitk.GetImageFromArray(image.type(torch.uint8)))

        writer.SetFileName("{}/{}.png".format(mask_dir, slice_n))
        writer.Execute(sitk.GetImageFromArray(mask))

    def generate_mask(self, segments :np.array) -> np.array :
        '''
        Returns a self suprevised mask of the given image.
        Needs to be run after window and imagify
        '''

        img_x_dim = segments.shape[0]
        img_y_dim = segments.shape[1]

        selected_pixels = self.super_pixels @ np.array([[img_x_dim, 0], [0, img_y_dim]])    # don't hard code image resolution
        selected_pixels = selected_pixels.astype('int32')

        selected_segments = [segments[tuple(sp)] for sp in selected_pixels]

        pre_mask = [segments == ss for ss in selected_segments]

        mask = pre_mask[0].type(torch.uint8)
        for i in range(1, len(pre_mask)):
            mask = torch.maximum(mask, (i + 1) * pre_mask[i].type(torch.uint8))

        return mask # convert to int mask