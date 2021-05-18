from torchvision import transforms
import albumentations as A

import sys
sys.path.append('.')

from data.CTDataSet import CTDicomSlicesFelzSaving
from data.CustomTransforms import Window, Imagify, MinDimension
from constants import Constants
from run_model import get_dl_workers
from data.DatasetPreprocessor import DatasetPreprocessor

'''
This script uses Felzenszwalb segmentation to crop images and then
also uses Felzenszwalb segementation on the cropped images to create
masks with super pixel segments. Saves all to given file.
'''

WL = 50
WW = 200
size = 256

def create_dataset():
    dataset = Constants.ct_only_filtered2
    dcm_list = CTDicomSlicesFelzSaving.generate_file_list(dataset, dicom_glob='/*/*/*.dcm')
    
    prep = transforms.Compose([Window(WL, WW), Imagify(WL, WW)])
    resize_tsfm = MinDimension(size)
    tsfm = A.CenterCrop(size, size, always_apply=True, p=1.0)

    ctds = CTDicomSlicesFelzSaving(dcm_list, preprocessing=prep, transform=tsfm, resize_transform = resize_tsfm, felz_crop=True)

    return ctds

if __name__ == '__main__':
    n_workers = get_dl_workers()

    ds = create_dataset()
    dp = DatasetPreprocessor(ds, Constants.ct_only_cleaned_resized, num_workers = n_workers,
                             shuffle = False)
                             
    dp.process_dataset()
