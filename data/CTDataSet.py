from torch.utils.data import Dataset
import glob
import os
import SimpleITK as sitk
import numpy as np
import random
import cv2
from skimage.segmentation import felzenszwalb
from pathlib import Path

from holdout import read_list, write_list

class CTDicomSlices(Dataset):
    '''
    Loads CT stacks and their segmentations
    '''
    def __init__(self, dcm_file_list :list, transform = None, img_and_mask_transform = None,
                shuffle = False, preprocessing = None, same_image_all_channels = False,
                n_surrounding :int = 1, trim_edges :bool = False, self_supervised_mask = False,
                resize_transform = None, save_masks_path :str = None, dirs_to_dcm :int = 1):
        '''
        Initializes a new CTDicomSlices

        Parameters:
        dcm_file_list: list of .dcm files containing slices of all patiences in dataset
        transform: albumentation to transform input slices
        img_and_mask_transform: albumentation transform both input slices and mask
        shuffle: whether or not to shuffle the dataset
        preprocessing: function to transform input slices
        same_image_all_channels: all channels use the same slice
        n_surrounding: number of slices to take per input. Each item will have n_surrounding * 2 + 1 slices
        trim_edges: trims columns and rows that are empty. May result in different resolutions for each image.
                    However, a given stack of surrounding slices will all have the same size - each trimmed by
                    the min amount.
        resize_transform: the transform to resize an image and mask. If self supervised, resized transform
                          should resize image only. Otherwise, it should resize both image and mask.
        save_mask_path: saves all masks to the given dir. This is useful if creating self-supervised masks.
                        if None, masks are not saved.
        dirs_to_dcm: specifies how many dirs need to be traversed from dataset directory to get to DICOMs
                     for main dataset, this should be 2. for pre-training dataset, this should be 1.
                     this is used only for saving masks
        '''
        # DICOM files
        self.dcm_list = dcm_file_list.copy()
        if shuffle:
            random.shuffle(self.dcm_list)

        self.transform = transform
        self.img_and_mask_transform = img_and_mask_transform
        self.preprocessing = preprocessing
        self.same_image_all_channels = same_image_all_channels
        self.n_surrounding = n_surrounding
        self.trim_edges = trim_edges
        self.self_supervised_mask = self_supervised_mask
        self.resize_transform = resize_transform
        self.save_masks_path = save_masks_path
        self.dirs_to_dcm = dirs_to_dcm

    def __getitem__(self, idx):
        '''
        Returns a tuple of image and mask.
        Combines all GTVp and GTVn masks into 1 mask.
        '''
        img_path = self.dcm_list[idx]
        slice_n = os.path.basename(img_path)[0:-4]
        slice_n = int(slice_n)

        slices = self.get_n_slices(img_path, slice_n, self.n_surrounding)
        slices = np.moveaxis(slices, 0, -1)

        if self.preprocessing is not None:
            # Typically window
            slices = self.preprocessing(slices)

        if self.self_supervised_mask:
            if self.trim_edges:
                slices = self.crop_image_only_outside(slices)

            if self.resize_transform is not None:
                slices = self.resize_transform(image=slices)['image']

            if self.transform is not None:
                # Typically gassian noise, scale, rotate
                slices = self.transform(image=slices)['image']

            mask = self.get_felzenszwalb(slices)

            if self.img_and_mask_transform is not None:
                # Not used
                print("WARNING: Image and mask transform is used in a self-supervised setting. This will be ignored.")
                #sample = self.img_and_mask_transform(image=slices, mask=mask)
                #slices, mask = sample['image'], sample['mask']

        else:
            mask = self.get_mask(img_path, slice_n)

            if self.trim_edges:
                slices, mask = self.crop_image_only_outside(slices, mask)

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

        if self.save_masks_path is not None:
            save_mask(img_path, slice_n)
            
        return slices.astype("float32"), mask.astype("float32"), img_path, slice_n

    def save_mask(self, img_path :str, slice_n :int):
        subdirs = []
        current_path = os.path.dirname(img_path)
        for i in range(self.dirs_to_dcm):
            current_path, current_dir = os.path.split(current_path)
            subdirs.insert(0, current_dir)

        save_path = "{}/{}/{}.dcm".format(self.save_masks_path, "/".join(subdirs) ,slice_n)


    def crop_image_only_outside(self, img :np.array, original_mask :np.array = None, tol :int = 0):
        '''
        Crops empty rows and columns on edge of images
        This should be run after windowing
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

        if original_mask is not None:
            return img[row_start:row_end,col_start:col_end], original_mask[row_start:row_end,col_start:col_end]
        else:
            return img[row_start:row_end,col_start:col_end]

    def get_mask(self, img_path :str, slice_n :int):
        dicoms_dir = os.path.dirname(img_path)
        pt_dir = os.path.dirname(dicoms_dir)

        mask_component_dirs = glob.glob(pt_dir + '/GTV*')

        aggregate_mask = None

        for c_dir in mask_component_dirs:
            mask_comp = sitk.GetArrayFromImage(sitk.ReadImage(c_dir + "/" + str(slice_n) + ".png"))
            mask_comp = mask_comp // 255      # Reduce it to 1 vs 0 instead of 255 vs 0

            if aggregate_mask is not None:
                aggregate_mask = np.maximum(aggregate_mask, mask_comp)
            else:
                aggregate_mask = mask_comp

        return aggregate_mask

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
        for e in empties:
            if e <= slice_n:
                imgs.insert(0, imgs[0])
            else:
                imgs.insert(-1, imgs[-1])

        return np.concatenate(imgs, axis=0)

    def __len__(self):
        return len(self.dcm_list)

    def generate_file_list(patient_dir :str, dicom_glob :str = '/*/dicoms/*.dcm'):
        return glob.glob(patient_dir + dicom_glob)

    def get_felzenszwalb(self, slices :np.array) -> np.array :
        '''
        Returns a self suprevised mask of the given image.
        Needs to be run after window and imagify
        '''
        rows, cols = slices.shape[0], slices.shape[1]

        mid_slice = slices.shape[2] // 2

        segments = felzenszwalb(slices[:,:,mid_slice], scale=150, sigma=0.7, min_size=5)

        selected_pixels = np.array([[5/16, 5/16], [5/16, 11/16], [11/16, 5/16], [11/16, 11/16]]) @ np.array([[rows, 0], [0, cols]])    # don't hard code image resolution
        selected_pixels = selected_pixels.astype('int32')

        selected_segments = [segments[tuple(sp)] for sp in selected_pixels]

        pre_mask = [segments == ss for ss in selected_segments]

        mask = np.logical_or.reduce(pre_mask)

        return (mask * 1).astype("uint8") # convert to int mask


class DatasetManager():
    def __init__(self, patient_dir :str, train :list, val :list, test :list):
        self.patient_dir = patient_dir
        self.train = train
        self.val = val
        self.test = test

    @classmethod
    def generate_train_val_test(cls, patient_dir :str, val_frac :float = 0.112, test_frac :float = 0.112):
        '''
        Returns a DatasetManager containing the given split of training, validation and test sets on the 
        patient level.
        '''
        pt_list = glob.glob("{}/*".format(patient_dir))
        pt_list = [os.path.basename(p) for p in pt_list]

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

    def get_dicoms(self, dicom_glob :str = 'dicoms/*.dcm'):
        '''
        Returns a tuple of three lists, containing the dicom files for train, val and test in that order
        '''
        train = self._get_dicoms(self.train)
        val = self._get_dicoms(self.val)
        test = self._get_dicoms(self.test)

        return (train, val, test)

    def save_lists(self, location :str, train :str = "train.txt", val :str = "val.txt", test :str = "test.txt"):
        '''
        Saves the list of patients to the given location using the given file names
        '''
        write_list("{}/{}".format(location, train), self.train)
        write_list("{}/{}".format(location, val), self.val)
        write_list("{}/{}".format(location, test), self.test)