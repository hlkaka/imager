from torch.utils.data import Dataset
import glob
import os
import SimpleITK as sitk
import numpy as np
import random
import cv2

from holdout import read_list, write_list

class CTDicomSlices(Dataset):
    '''
    Loads CT stacks and their segmentations
    '''
    def __init__(self, dcm_file_list :list, transform = None, img_and_mask_transform = None,
                shuffle = False, preprocessing = None):
        # DICOM files
        self.dcm_list = dcm_file_list.copy()
        if shuffle:
            random.shuffle(self.dcm_list)

        self.transform = transform
        self.img_and_mask_transform = img_and_mask_transform
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        '''
        Returns a tuple of image and mask.
        Combines all GTVp and GTVn masks into 1 mask.
        '''
        img_path = self.dcm_list[idx]
        slice_n = os.path.basename(img_path)[0:-4]
        slice_n = int(slice_n)

        slices = self.get_surroundings(img_path, slice_n)
        slices = np.moveaxis(slices, 0, -1)

        mask = self.get_mask(img_path, slice_n)

        if self.preprocessing is not None:
            slices = self.preprocessing(slices)

        if self.transform is not None:
            slices = self.transform(slices)

        if self.img_and_mask_transform is not None:
            sample = self.img_and_mask_transform(image=slices, mask=mask)
            slices, mask = sample['image'], sample['mask']

        return slices, mask.astype("float32"), img_path, slice_n

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

        # No longer required. Rotation is done in dcm creation.
        #aggregate_mask = np.rot90(aggregate_mask)
        #aggregate_mask = np.flipud(aggregate_mask)

        return aggregate_mask

    def get_surroundings(self, img_path :str, slice_n :int):
        '''
        Gets the slice and 1 before and after
        If 1 before or after does not exist, returns the slice itself
        Returns a numpy array of shape: [3, 512, 512] with position [1, :, :] being the desired slice
        '''
        dicoms_dir = os.path.dirname(img_path)
        
        prev_path = dicoms_dir + "/" + str(slice_n - 1) + ".dcm"
        next_path = dicoms_dir + "/" + str(slice_n + 1) + ".dcm"

        img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))

        if os.path.isfile(prev_path):
            prev_img = sitk.GetArrayFromImage(sitk.ReadImage(prev_path))
        else:
            prev_img = img

        if os.path.isfile(next_path):
            next_img = sitk.GetArrayFromImage(sitk.ReadImage(next_path))
        else:
            next_img = img

        return np.concatenate((prev_img, img, next_img), axis = 0)

    def __len__(self):
        return len(self.dcm_list)

    def generate_file_list(patient_dir :str, dicom_glob :str = '/*/dicoms/*.dcm'):
        return glob.glob(patient_dir + dicom_glob)


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