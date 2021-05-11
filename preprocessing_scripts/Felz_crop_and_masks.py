from torchvision import transforms

import sys
sys.path.append('data/')
sys.path.append('.')

from CTDataSet import CTDicomSlicesFelzSaving
from CustomTransforms import Window
from CustomTransforms import Imagify
from constants import Constants

from DatasetPreprocessor import DatasetPreprocessor

'''
This script uses Felzenszwalb segmentation to crop images and then
also uses Felzenszwalb segementation on the cropped images to create
masks with super pixel segments. Saves all to given file.
'''

def create_dataset():
    dataset = Constants.ct_only_filtered2
    dcm_list = CTDicomSlicesFelzSaving.generate_file_list(dataset, dicom_glob='*/*/*/*.dcm')
    
    prep = transforms.Compose([Window(50, 250), Imagify(50, 250)])
    ctds = CTDicomSlicesFelzSaving(dcm_list, preprocessing=prep, felz_crop=True)

    return ctds

if __name__ == '__main__':
    n_workers = input("Enter number of works (usually = num CPU cores) - no error checking:")
    n_workers = int(n_workers)

    ds = create_dataset()
    dp = DatasetPreprocessor(ds, Constants.ct_only_cleaned, num_workers = n_workers,
                             shuffle = True)
                             
    dp.process_dataset()
