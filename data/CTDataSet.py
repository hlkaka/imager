from torch.utils.data import Dataset
import glob
import os
from os.path import basename, dirname
import SimpleITK as sitk
import numpy as np
import random
from skimage.segmentation import felzenszwalb
import albumentations as A
import sys
from tqdm import trange
from scipy.spatial.distance import cdist
import itertools
import torch

sys.path.append('.')
from constants import Constants
from data.holdout import read_list, write_list

class CTDicomSlicesMaskless(Dataset):
    '''
    Loads CT stacks
    '''

    SMALL_DS_DICOM_GLOB = '/*/dicoms/*.dcm'
    LARGE_DS_DICOM_GLOB = '/*/*/*.dcm'

    def __init__(self, dcm_file_list :list, transform = None, preprocessing = None,
                n_surrounding :int = 1, trim_edges :bool = False, resize_transform = None, same_img_all_channels = False):
        '''
        Initializes a new CTDicomSlices

        Parameters:
        dcm_file_list: list of .dcm files containing slices of all patiences in dataset
        transform: albumentation to transform input slices
        shuffle: whether or not to shuffle the dataset
        preprocessing: function to transform input slices
        n_surrounding: number of slices to take per input. Each item will have n_surrounding * 2 + 1 slices
        trim_edges: trims columns and rows that are empty. May result in different resolutions for each image.
                    However, a given stack of surrounding slices will all have the same size - each trimmed by
                    the min amount.
        '''
        # DICOM files
        self.dcm_list = dcm_file_list.copy()

        self.transform = transform
        self.preprocessing = preprocessing
        self.n_surrounding = n_surrounding
        self.trim_edges = trim_edges
        self.resize_transform = resize_transform
        self.same_img_all_channels = same_img_all_channels

    def get_n_slices(self, img_path :str, slice_n :int, surrounding :int):
        '''
        Gets n slices from the given image path.
        If surrounding = 0: gets, only a single slice
        If surrounding > 0: gets surrounding many slides before and after
            i.e. total number of slides is surrounding * 2 + 1
        '''
        dicoms_dir = os.path.dirname(img_path)

        imgs = []
        empties = [] # This will hold hypothetical slice number of images that don't exist
        
        if self.same_img_all_channels:
            slice_path = "{}/{}.dcm".format(dicoms_dir, slice_n)
            if os.path.isfile(slice_path):
                imgs.append(sitk.GetArrayFromImage(sitk.ReadImage(slice_path)))
        else:
            # +1 because range(start, stop) does not include stop
            for i in range(slice_n - surrounding, slice_n + surrounding + 1):
                slice_path = "{}/{}.dcm".format(dicoms_dir, i)
                if os.path.isfile(slice_path):
                    imgs.append(sitk.GetArrayFromImage(sitk.ReadImage(slice_path)))
                else:
                    empties.append(i)
        
        # Order is important
        # In above loop, we go from lowest n slice to the highest
        # Here, we replicate the earliest available slice in beginning and the last available slice in end
        if self.same_img_all_channels:
            for i in range(surrounding):
                imgs.insert(0, imgs[0])
                imgs.insert(-1, imgs[-1])

        else:
            for e in empties:
                if e <= slice_n:
                    imgs.insert(0, imgs[0])
                else:
                    imgs.insert(-1, imgs[-1])

        return np.concatenate(imgs, axis=0)

    def get_path_slice_num(self, idx :int):
        ''' Gets image path and slice number from given idx '''
        img_path = self.dcm_list[idx]
        slice_n = os.path.basename(img_path)[0:-4]
        slice_n = int(slice_n)

        return img_path, slice_n

    def get_images(self, idx :int):
        '''
        Reads the images
        If single_image is True, ignores self.n_surrounding
        '''
        img_path, slice_n = self.get_path_slice_num(idx)

        slices = self.get_n_slices(img_path, slice_n, self.n_surrounding)
        slices = np.moveaxis(slices, 0, -1)

        return slices, img_path, slice_n
        
    def apply_all_transforms(self, slices, exclude_prep = False):
        if self.preprocessing is not None and not exclude_prep:
            # Typically window
            slices = self.preprocessing(slices)

        if self.trim_edges:
            row_start, row_end, col_start, col_end = self.crop_image_only_outside(slices)
            slices = slices[row_start:row_end,col_start:col_end]

        if self.resize_transform is not None:
            slices = self.resize_transform(image=slices)['image']

        if self.transform is not None:
            # Typically gassian noise, scale, rotate
            slices = self.transform(image=slices)['image']

        return slices


    def __getitem__(self, idx):
        '''
        Returns a tuple of image, img_path and slice_n.
        Shape of slices is H x W x # slices
        '''
        slices, img_path, slice_n = self.get_images(idx)
        slices = self.apply_all_transforms(slices)
            
        return slices.astype("float32"), img_path, slice_n

    def crop_image_only_outside(self, img :np.array, tol :int = 0):
        '''
        Crops empty rows and columns on edge of images
        This should be run after windowing
        Returns coordinates for cropping

        Code taken from here: https://codereview.stackexchange.com/questions/132914/crop-black-border-of-image-using-numpy
        '''
        # img is 2D or 3D image data
        # tol  is tolerance
        mask = img>tol
        if img.ndim==3:
            mask = mask.all(2)

        m,n = mask.shape
        mask0,mask1 = mask.any(0),mask.any(1)
        col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
        row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()

        return row_start, row_end, col_start, col_end

    def __len__(self):
        return len(self.dcm_list)

    def calculate_ds_mean_std(self):
        '''
        Returns the mean and standard deviation for pixel values.
        Applies the supplied transformations.
        '''
        # Code from __getitem__ is replicated here to resolve self. vs super(). considerations
        # in inherited classes and to ensure single slices are returned each time
        sum_means = 0
        sum_stds = 0

        print('Calculating Dataset Means')

        for idx in trange(self.__len__()):
            slices, _, _ = self.get_images(idx)
            slices = self.apply_all_transforms(slices)
            
            sum_means = sum_means + np.mean(slices)
            sum_stds = sum_stds + np.mean(slices)

        return sum_means/self.__len__(), sum_stds/self.__len__()

    @staticmethod
    def generate_file_list(ds_dir :str, dicom_glob :str = '/*/dicoms/*.dcm'):
        return glob.glob(ds_dir + dicom_glob)

class CTDicomSlices(CTDicomSlicesMaskless):
    '''
    Loads CT stacks and their existing segmentations
    '''
    def __init__(self, dcm_file_list :list, transform = None, img_and_mask_transform = None,
                preprocessing = None, n_surrounding :int = 1,
                trim_edges :bool = False, resize_transform = None, mask_is_255 = True, same_img_all_channels = False):
        '''
        Initializes a new CTDicomSlices

        Parameters:
        dcm_file_list: list of .dcm files containing slices of all patiences in dataset
        transform: albumentation to transform input slices
        img_and_mask_transform: albumentation transform both input slices and mask
        preprocessing: function to transform input slices
        n_surrounding: number of slices to take per input. Each item will have n_surrounding * 2 + 1 slices
        trim_edges: trims columns and rows that are empty. May result in different resolutions for each image.
                    However, a given stack of surrounding slices will all have the same size - each trimmed by
                    the min amount.
        resize_transform: the transform to resize an image and mask. If self supervised, resized transform
                          should resize image only. Otherwise, it should resize both image and mask.
        mask_is_255: if True, the mask pixel values are divided by 255 to get class values. I.e. foreground is 255 and background is 0.
                     if False, the mask pixel values are kept as is to get class values. I.e. foreground is 1,2,3 and background is 0.
        '''
        super().__init__(dcm_file_list, transform=transform, preprocessing=preprocessing,
                         n_surrounding=n_surrounding, trim_edges=trim_edges, resize_transform=resize_transform, same_img_all_channels = same_img_all_channels)

        self.img_and_mask_transform = img_and_mask_transform
        self.mask_is_255 = mask_is_255
        
    def apply_all_transforms_with_masks(self, slices, mask):
        if self.preprocessing is not None:
            # Typically window
            slices = self.preprocessing(slices)

        if self.trim_edges:
            row_start, row_end, col_start, col_end = self.crop_image_only_outside(slices)
            slices = slices[row_start:row_end,col_start:col_end]
            mask = mask[row_start:row_end,col_start:col_end]

        if self.resize_transform is not None:
            # Typically scale, rotate, etc
            sample = self.resize_transform(image=slices, mask=mask)
            slices, mask = sample['image'], sample['mask']

        if self.transform is not None:
            # Typically gassian noise
            slices = self.transform(image=slices)['image']

        if self.img_and_mask_transform is not None:
            # Typically scale, rotate, etc
            sample = self.img_and_mask_transform(image=slices, mask=mask)
            slices, mask = sample['image'], sample['mask']

        return slices, mask


    def __getitem__(self, idx):
        '''
        Returns a tuple of image, mask, image path and slice_n.
        Combines all GTVp and GTVn masks into 1 mask.
        Shape of slices is H x W x # slices
        '''
        slices, img_path, slice_n = super().get_images(idx)
        mask = self.get_mask(img_path, slice_n)

        slices, mask = self.apply_all_transforms_with_masks(slices, mask)
            
        return slices.astype("float32"), mask.astype("long"), img_path, slice_n

    def get_mask(self, img_path :str, slice_n :int):
        dicoms_dir = os.path.dirname(img_path)
        pt_dir = os.path.dirname(dicoms_dir)

        mask_component_dirs = glob.glob(pt_dir + '/GTV*')

        aggregate_mask = None

        for c_dir in mask_component_dirs:
            mask_comp = sitk.GetArrayFromImage(sitk.ReadImage(c_dir + "/" + str(slice_n) + ".png"))
            if self.mask_is_255:
                mask_comp = mask_comp // 255      # Reduce it to 1 vs 0 instead of 255 vs 0

            if aggregate_mask is not None:
                aggregate_mask = np.maximum(aggregate_mask, mask_comp)
            else:
                aggregate_mask = mask_comp

        return aggregate_mask

class CTDicomSlicesFelzenszwalb(CTDicomSlices):
    '''
    Loads CT stacks and creates felzenswalb segmentations
    '''
    def __init__(self, dcm_file_list :list, transform = None, preprocessing = None, \
                n_surrounding :int = 1, trim_edges :bool = False, resize_transform = None, \
                super_pixels = None, felz_params :dict = None, felz_crop = False, same_img_all_channels = False):
        '''
        Initializes a new CTDicomSlices

        Parameters:
        dcm_file_list: list of .dcm files containing slices of all patiences in dataset
        transform: albumentation to transform input slices
        preprocessing: function to transform input slices. This is the only transform which occurs before felz crop
        n_surrounding: number of slices to take per input. Each item will have n_surrounding * 2 + 1 slices
        trim_edges: trims columns and rows that are empty. May result in different resolutions for each image.
                    However, a given stack of surrounding slices will all have the same size - each trimmed by
                    the min amount.
        resize_transform: the transform to resize an image and mask. If self supervised, resized transform
                          should resize image only. Otherwise, it should resize both image and mask.
        super_pixels: the pixels from which the segments will be selected. Needs to be [0, 1]
        felz_params: dictionary for Felzenszwalb algorithm 
             default is {'scale':150, 'sigma':0.6, 'min_size':50}
             these are used fro the mask, but not for cropping
        felz_crop: this is used to crop the image using the Felz background segment
        '''
        # img_and_mask_transform was removed as scale/rotate transforms would remove the learnable information
        # regarding the positions of super pixels in the image

        super().__init__(dcm_file_list, transform = transform, img_and_mask_transform = None,
                        preprocessing=preprocessing, n_surrounding=n_surrounding, trim_edges=trim_edges,
                        resize_transform=resize_transform, same_img_all_channels = same_img_all_channels)
    
        if super_pixels is None:
            self.super_pixels = np.array([[5/16, 5/16], [5/16, 11/16], [1/2, 1/2], [11/16, 5/16], [11/16, 11/16]])
        else:
            self.super_pixels = super_pixels

        self.felz_params = felz_params
        if self.felz_params is None:
            self.felz_params = Constants.default_felz_params

        self.felz_crop = felz_crop

    def get_felzenszwalb(self, slices :np.array, felz_params :dict) -> np.array :
        '''
        Returns an array of felzenswalb segments
        Needs to be run after window and imagify
        Returns (H x W) of the main slice
        '''
        mid_slice = slices.shape[2] // 2

        segments = felzenszwalb(slices[:,:,mid_slice], scale=150, sigma=0.6, min_size=50)

        return (segments).astype("uint8") # convert to int mask

    def get_mask(self, segments, single_foreground :bool = True):
        # Initial implementation is in test_CTDataSet in function get_felzenszwalb
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

        mask = pre_mask[0].astype('uint8')
        for i in range(1, len(pre_mask)):
            if single_foreground:
                # generates a single segment
                mask = np.maximum(mask, pre_mask[i].astype('uint8'))
            else:
                # generates 5 different segments
                mask = np.maximum(mask, (i + 1) * pre_mask[i].astype('uint8'))

        return mask, selected_pixels # convert to int mask

    def apply_crops(self, slices):
        '''
        Performs all crops.
        If return_preprocessed is true, the function will return a cropped pre-processed image
        Otherwise, it will return a non-pre_processed image
        '''
        if self.preprocessing is not None:
            # Typically window
            slices_prep = self.preprocessing(slices)
        else:
            slices_prep = slices

        segments_for_cropping = self.get_felzenszwalb(slices_prep, Constants.default_felz_params)
        segments_for_cropping = np.expand_dims(segments_for_cropping, 2)

        slices, _ = self.segmented_crop(slices, segments_for_cropping)
        return slices

    def __getitem__(self, idx):
        '''
        Order is: preprocess, do felz_crop (if needed), do all other transforms,
                  re-segment transformed image, create mask
        Shape of slices is H x W x # slices
        '''
        slices, img_path, slice_n = super().get_images(idx)

        if self.felz_crop:
            slices = self.apply_crops(slices)
        
        slices = self.apply_all_transforms(slices)

        # Segment again after the image is transformed properly
        # This is important as transforms can include resizing
        segments_final = self.get_felzenszwalb(slices, self.felz_params)
        mask, _ = self.get_mask(segments_final)

        return slices.astype("float32"), mask.astype("long"), img_path, slice_n

    def get_crops(self, segments, dim, step = 1, buffer = 20, bg_segment=0):
        '''
        Uses the background segment of felzenswalb to strip the vertical and horizontal edges of images.
        segments: felzenswalb segmentation
        bg_segment: usually 0. The segment label assigned to background.
        dim: dimension to act on. 1 strips top and bottom, 0 strips left and right.
        step: how many pixels to scan every step. Default is 3
        buffer: how many pixels to leave before stripping
        return: the coordinates of the foreground pixels (i.e. noncropped pixels)
        along the dimension supplied
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

    def segmented_crop(self, slices, segments):
        '''
        Crops the slices and segments based on Felzenswalb background segment
        input must be (H x W x # slices)
        '''
        v_dims = self.get_crops(segments, 1)
        h_dims = self.get_crops(segments, 0)

        if v_dims is not None:
            slices = slices[v_dims[0]:v_dims[1], :, :]
            segments = segments[v_dims[0]:v_dims[1], :, :]
        
        if h_dims is not None:
            slices = slices[:, h_dims[0]:h_dims[1], :]
            segments = segments[:, h_dims[0]:h_dims[1], :]

        return slices, segments

class CTDicomSlicesFelzSaving(CTDicomSlicesFelzenszwalb):
    '''
    Inherits the Fezenszwalb class to return single images only
    but with meta data. This will be used to save them to disk
    after reformatting. Predominantly for DatasetPreprocessor
    The images will be transformed to generate segments and mask, but otherwise
    this class is different from CTDicomSlicesFelzenszwalb in that it returns
    the image without pre_processing. Only the resize transform is applied
    to the returned image.
    '''
    def __init__(self, dcm_file_list :list, transform = None, preprocessing = None, \
                trim_edges :bool = False, resize_transform = None, \
                super_pixels = None, felz_params :dict = None, felz_crop = False,
                preprocess_output = False, single_foreground = True):
        '''
        Preprocess then segment then crop.
        The resultant image can skip pre_processing (i.e. skip windowing) if pre_process_output is False
        Otherwise, windowed images will be returned
        '''
        # img_and_mask_transform was removed as scale/rotate transforms would remove the learnable information
        # regarding the positions of super pixels in the image

        super().__init__(dcm_file_list, transform = transform, preprocessing=preprocessing,
                        n_surrounding=0, trim_edges=trim_edges, resize_transform=resize_transform,
                        super_pixels=super_pixels, felz_params=felz_params, felz_crop=felz_crop)

        self.preprocess_output = preprocess_output
        self.single_foreground = single_foreground

    def image_with_metadata(self, img_path :str, slice_n :int):
        '''
        Gets the slice from the given path. Also gets DICOM metadata
        Image is H x W x 1   --- 1 = # slices
        '''
        dicoms_dir = os.path.dirname(img_path)
    
        slice_path = "{}/{}.dcm".format(dicoms_dir, slice_n)

        if os.path.isfile(slice_path):
            image_file = sitk.ReadImage(slice_path)
            image = sitk.GetArrayFromImage(image_file)
        
        # GetArrayFromImage will have channel = 1. Move it to end for compatibility
        image = np.moveaxis(image, 0, -1)

        # Get meta data
        metadata = {}
        for k in image_file.GetMetaDataKeys():
            metadata[k] = image_file.GetMetaData(k)

        return image, metadata

    def __getitem__(self, idx):
        '''
        Returns a single image with its mask & metadata dictionary.
        Order is: preprocess, do felz_crop (if needed), do all other transforms,
                  re-segment transformed image, create mask
        Shape of slices is H x W x # slices
        '''

        '''
        1. Window
        2. Segment-1
        3. Felz crop original image
        4. Transform Felz cropped original
        5. Segment-2
        6. Generate mask
        '''
        img_path, slice_n = self.get_path_slice_num(idx)
        image, metadata = self.image_with_metadata(img_path, slice_n)

        # Performs preprocessing for cropping. However, returns a cropped unpreprocessed image
        if self.felz_crop:
            image = self.apply_crops(image)   # Does 1-3

        image_transformed = self.apply_all_transforms(image)   # Does 4
        segments = self.get_felzenszwalb(image_transformed, self.felz_params) # Does 5
        mask, super_pixels = self.get_mask(segments, single_foreground = self.single_foreground) # Does 6

        # Finally, get everything on image except prep
        image = self.apply_all_transforms(image, exclude_prep=True)

        return image.astype("float32"), mask.astype("long"), \
                img_path, slice_n, segments.astype("float32"), metadata, super_pixels

class CTDicomSlicesJigsaw(CTDicomSlicesMaskless):
    '''
    Loads CT stacks and creates jigsaw puzzles of their slices
    '''
    def __init__(self, dcm_file_list :list, transform = None, preprocessing = None,
                trim_edges :bool = False, resize_transform = None, sqrt_n_jigsaw_pieces :int = 3,
                min_img_size :int = 225, tile_size :int = 64,
                return_tile_coords = False, n_shuffles_per_image = 36,
                perm_path=None, num_perms = 1000, same_img_all_channels = False, tile_normalize = True):

        '''
        min_img_size: dimension of the grid from which tiles will be taken
        this must be the minimum image size. otherwise, smaller images will
        be resized up

        tile_size: the size of each tile. typically smaller than grid sections
        normalize_tiles: independently normalize each tile separately
        return_tile_coords: returns the coords of selected tiles, for result printing
        '''

        # Make sure the puzzle size is divisble by number of puzzle pieces
        assert min_img_size % sqrt_n_jigsaw_pieces == 0

        super().__init__(dcm_file_list, transform=transform, preprocessing=preprocessing,
                        n_surrounding=0, trim_edges=trim_edges, resize_transform=resize_transform, same_img_all_channels = same_img_all_channels)
        
        self.snjp = sqrt_n_jigsaw_pieces
        self.min_img_size = 225
        self.tile_size = tile_size
        self.return_tile_coords = return_tile_coords
        self.n_shuffles_per_image = n_shuffles_per_image
        self.tile_normalize = tile_normalize

        if num_perms is not None:
            self.load_permutations(perm_path, num = num_perms, replace = False)
        else:
            self.perms = None

    def ensure_min_size(self, image):
        ''' Ensures each dimension of the image is >= self.min_img_size '''
        dims = image.shape[0:2]

        if dims[0] < self.min_img_size or dims[1] < self.min_img_size:
            # image is too small
            ratio = self.min_img_size / min(dims)
            resizer = A.RandomScale([ratio, ratio], always_apply=True, p=1.0)
            image = resizer(image=image)['image']

        return image

    def pick_tile_top_left_pixel(self, bg_size, fg_size :int):
        '''
        Uniformly samples a top-left pixel for the foreground from the given background
        Top left pixel needs to fall in the rectange between (0, 0) and (background_x - forground_x, background_y - foreground_y)
        + 1 because randint does not include the upper bound
        '''
        # If bg_size is int, convert to 2-dimensional array
        # No need to bother with fg_size, as numpy will take care of that as long as one is an array
        if isinstance(bg_size, int) or (isinstance(bg_size, type(np.array)) and bg_size.shape == 1):
            bg_size = np.array([bg_size, bg_size])
        
        top_left_rectangle = bg_size - fg_size + 1
        
        top_left_pixel = np.array([np.random.randint(0, high=top_left_rectangle[0]), np.random.randint(0, high=top_left_rectangle[1])])

        return top_left_pixel
    
    def pick_grid_top_left_pixel(self, bg_size, fg_size :int):
        '''
        Uniformly samples a top-left pixel for the foreground from the given background
        Top left pixel needs to fall in the rectange between (0, 0) and (background_x - forground_x, background_y - foreground_y)
        + 1 because randint does not include the upper bound
        '''
        # If bg_size is int, convert to 2-dimensional array
        # No need to bother with fg_size, as numpy will take care of that as long as one is an array
        if isinstance(bg_size, int) or (isinstance(bg_size, type(np.array)) and bg_size.shape == 1):
            bg_size = np.array([bg_size, bg_size])
        
        top_left_rectangle = np.floor((bg_size - fg_size + 1) / 2)
        
        rectangle_half_size = 16   # magic number

        top_left_pixel = np.array([np.random.randint(top_left_rectangle[0] - rectangle_half_size, high=top_left_rectangle[1] + rectangle_half_size),
                                   np.random.randint(top_left_rectangle[0] - rectangle_half_size, high=top_left_rectangle[1] + rectangle_half_size)])

        return top_left_pixel
    

    def pick_puzzle_coords(self, image):
        '''
        Randomly selects a location of the puzzle crop from the image.
        '''
        dims = image.shape[0:2] 
        top_left_pixel = self.pick_grid_top_left_pixel(dims, self.min_img_size)

        return top_left_pixel

    def get_tiles(self, image, grid_top_left_pixel):
        '''
        Gets an array of all the tiles in the puzzle
        (snjp x tile_size x tile_size)        
        '''
        tiles = np.empty([self.snjp ** 2, self.tile_size, self.tile_size])

        grid_stride = int(self.min_img_size / self.snjp)

        if self.return_tile_coords:
            coords = []
        else:
            coords = None

        for i in range(self.snjp):
            for j in range(self.snjp):
                tile_top_left_main = np.array([i * grid_stride, j * grid_stride]) + grid_top_left_pixel

                tile_top_left = self.pick_tile_top_left_pixel(grid_stride, self.tile_size) + tile_top_left_main
                tile_bottom_right = tile_top_left + self.tile_size

                if self.return_tile_coords:
                    coords.append(tile_top_left)

                tile_data = image[tile_top_left[0]:tile_bottom_right[0], tile_top_left[1]:tile_bottom_right[1]]

                print("tile top left: {} -- tile bottom right: {}\n".format(tile_top_left, tile_bottom_right))

                np.copyto(tiles[i * self.snjp + j], tile_data.squeeze())

        return tiles, coords

    def generate_tiles(self, image):
        ''' Generates the jigsaw blocks '''
        image = self.ensure_min_size(image)
        dims = image.shape[0:2]
        
        grid_top_left = self.pick_grid_top_left_pixel(np.array(dims), self.min_img_size)
        # coords is for debugging
        tiles, coords = self.get_tiles(image, grid_top_left)

        return tiles, coords

    def __getitem__(self, idx):
        image, img_path, slice_n = super().__getitem__(idx)
        # coords is for debugging
        tiles, coords = self.generate_tiles(image)

        # normalize each tile
        tile_means = np.mean(tiles, axis=(1, 2))
        tile_stds = np.std(tiles, axis=(1, 2))

        if self.tile_normalize:
            tile_stds [tile_stds < 1e-1] = 1e-1
            tiles = (tiles - tile_means[:,None,None]) / tile_stds[:,None,None]
        
        # shuffle tiles
        # return permutation index as label
        # repeat n_shuffles_per_image times
        all_tiles = []
        all_labels = []

        for i in range(0, self.n_shuffles_per_image):
            if self.perms is not None:
                random_perm = random.randint(0, len(self.perms) - 1)
                all_labels.append(random_perm)
                all_tiles.append(tiles[self.perms[random_perm]])
            else:
                random_perm = np.arange(self.snjp ** 2)
                np.random.shuffle(random_perm)
                all_labels.append(random_perm)
                all_tiles.append(tiles[random_perm])

        all_tiles = np.stack(all_tiles, axis=0)
        all_labels = np.array(all_labels)

        return image, img_path, slice_n, tiles, coords, all_tiles.astype("float32"), all_labels.astype("long")

    def generate_permutations(self, num):
        '''
        Generates n number of n_tiles x n_tiles permutations with maximum Hamming distance between them
        Code is from https://github.com/bbrattoli/JigsawPuzzlePytorch/blob/master/select_permutations.py
        '''
        n_tiles = self.snjp ** 2

        P_hat = np.array(list(itertools.permutations(list(range(n_tiles)), n_tiles)))
        n = P_hat.shape[0]
        
        for i in trange(num):
            if i==0:
                j = np.random.randint(n)
                P = np.array(P_hat[j]).reshape([1,-1])
            else:
                P = np.concatenate([P,P_hat[j].reshape([1,-1])],axis=0)
            
            P_hat = np.delete(P_hat,j,axis=0)
            D = cdist(P,P_hat, metric='hamming').mean(axis=0).flatten()
            
            j = D.argmax()
        
        return P

    def load_permutations(self, perm_path :str, num :int = 1000, replace = False):
        '''
        If the given file exists, loads their permutations. If the file contains more perms than required, selects only the first num
        If it does not (or any error exists), then creates new (num) permutations and saves them.
        If replace is true, then replaces existing permutations anyway.
        '''
        if perm_path is not None and replace == False and os.path.exists(perm_path):
            try:
                self.perms = np.load(perm_path)
                assert(len(self.perms) >= num)
                if (len(self.perms) > num):
                    self.perms = self.perms[0:num]
                print('Permutations file was read at {}'.format(perm_path))
                return
            except IOError as err:
                print('Permutations file cannot be read at {}'.format(perm_path))
        
        print('New {} permutations will be created.'.format(num))

        self.perms = self.generate_permutations(num)

        if perm_path is not None:
            np.save(perm_path, self.perms)
            print('New permutations were saved at {}.'.format(perm_path))

        return

def jigsaw_training_collate(tuples):
    '''
    This custom function stacks tiles such that get each tile as its own image in a batch
    Instead of adding another indexing dimension
    i.e. changes (batch_size, perms_per_image, :, :) to (batch_size x perms_per_image, :, :)
    Ignores all other parameters which are not necessary for training
    '''
    perms_per_image = len(tuples[0][5])
    batch_perms_tiles = [t[5] for t in tuples]
    tiles = torch.tensor(np.concatenate(batch_perms_tiles, axis=0))
    all_labels = [t[6] for t in tuples]
    all_labels = torch.tensor(np.concatenate(all_labels))

    return tiles, all_labels

class DatasetManager():
    def __init__(self, patient_dir :str, train :list, val :list, test :list):
        self.patient_dir = patient_dir
        self.train = train
        self.val = val
        self.test = test

    @classmethod
    def generate_train_val_test(cls, patient_dir :str, val_frac :float = 0.112, test_frac :float = 0.112, pretrain_ds=False):
        '''
        Returns a DatasetManager containing the given split of training, validation and test sets on the 
        patient level.
        '''
        ds_name = lambda p: basename(dirname(p))

        if pretrain_ds:
            pt_list = glob.glob("{}/*/*".format(patient_dir))
            pt_list = ["{}/{}".format(ds_name(p), basename(p)) for p in pt_list]
        else:
            pt_list = glob.glob("{}/*".format(patient_dir))
            pt_list = [basename(p) for p in pt_list]

        random.shuffle(pt_list)

        # number of patients
        n = len(pt_list)

        # Create splits
        pt_val = pt_list[:int(n * val_frac)]
        pt_test = pt_list[int(n * val_frac) : int(n * (val_frac + test_frac))]
        pt_train = pt_list[int(n * (val_frac + test_frac)) :]

        return cls(patient_dir, pt_train, pt_val, pt_test)

    @classmethod
    def load_train_val_test(cls, patient_dir :str, train_list :str, val_list :str, test_list :str):
        '''
        Returns a DatasetManager with the patient lists read from the given file paths.
        Needs a path to where data is and a path to a text file containing each list
        '''
        train = read_list(train_list)
        val = read_list(val_list)
        test = read_list(test_list)

        return cls(patient_dir, train, val, test)

    def _get_dicoms(self, pts, dicom_glob :str = 'dicoms/*.dcm'):
        '''
        Helper function to retrieve the dicom files from the given patient list
        '''
        flatten = lambda l: [item for sublist in l for item in sublist]

        dcms = [glob.glob("{}/{}/{}".format(self.patient_dir, p, dicom_glob)) for p in pts]

        return flatten(dcms)

    def get_dicoms(self, dicom_glob :str = 'dicoms/*.dcm', train_frac = 1.0):
        '''
        Returns a tuple of three lists, containing the dicom files for train, val and test in that order
        train_frac is the fraction of the dataset to be used
        '''
        if train_frac < 1.0:
            train_size = int(train_frac * len(self.train))
            train_list = random.sample(self.train, train_size)
        else:
            train_list = self.train

        train = self._get_dicoms(train_list, dicom_glob)
        val = self._get_dicoms(self.val, dicom_glob)
        test = self._get_dicoms(self.test, dicom_glob)

        return (train, val, test)

    def save_lists(self, location :str, train :str = "train.txt", val :str = "val.txt", test :str = "test.txt"):
        '''
        Saves the list of patients to the given location using the given file names
        '''
        write_list("{}/{}".format(location, train), self.train)
        write_list("{}/{}".format(location, val), self.val)
        write_list("{}/{}".format(location, test), self.test)