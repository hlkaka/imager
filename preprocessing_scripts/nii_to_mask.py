'''
Converts nii masks to jpeg masks
Creates two versions - tumor and nodes and combined
'''

import glob
import nibabel as nb
import numpy as np
import cv2
import os
import tqdm
import pydicom
from shutil import copyfile

def save_stack(stack, target_f):
    '''
    Saves a stack of masks
    '''
    # Move slice axis to first
    stack = np.moveaxis(stack, -1, 0)

    os.makedirs(target_f, exist_ok=True)

    # Save slice by slice
    for i in range(stack.shape[0]):
        # Dicoms are 1-based indices so add 1
        cv2.imwrite(target_f + "/" + str(i + 1) + ".png", stack[i, :, :] * 255)

def save_group(pt_path, nii_file, target_name):
    '''
    Saves all masks of a group for a given patient
    A group is either GTVp (primary) or GTVn (nodes)
    '''
    file_paths = glob.glob(pt_path + "/" + nii_file)
    files = map(nb.load, file_paths)

    pt_base = os.path.basename(pt_path)
    pt_t_f = separated_output_dir + "/" + pt_base + "/"

    # Save memory by loading only one at a time
    for i, f in enumerate(list(files)):
        # Volumes are 1-based indices
        save_stack(f.get_fdata(), pt_t_f + target_name + str(i + 1))

def copy_dcm(pt_path, dicom_dataset, target_f):
    pt_base = os.path.basename(pt_path)
    dcm_source = dicom_dataset + "/" + pt_base + "/"
    dcm_dest_f = target_f + "/" + pt_base + "/dicoms/"

    os.makedirs(dcm_dest_f, exist_ok=True)

    for dcm_fp in glob.glob(dcm_source + "*/*/*.dcm"):
        # If the file is called 1-1.dcm, then its the RT-STRUCT file not a DICOM image
        # In that case, skip
        if dcm_fp[-7:] == "1-1.dcm":
            continue

        with pydicom.dcmread(dcm_fp) as ds:
            slice_number = ds[0x0020, 0x0013].value
        
        slice_number = int(slice_number) # to get rid of any potential leading zeros

        if pt_base == 'HNSCC-01-0415' and slice_number > 239:
            print("Got to pt 415 and slice {}".format(slice_number))

        copyfile(dcm_fp, dcm_dest_f + str(slice_number) + ".dcm")

if __name__ == '__main__':
    dataset = "/home/hussam/nii_dataset/"
    dicom_dataset = "/mnt/e/HNSCC dataset/HNSCC"
    mask_p = "mask_GTVp*.nii.gz"
    mask_n = "mask_GTVn*.nii.gz"

    separated_output_dir = "/home/hussam/organized_dataset"

    pts = glob.glob(dataset + "*")

    for p in tqdm.tqdm(pts):
        save_group(p, mask_p, "GTVp")
        save_group(p, mask_n, "GTVn")
        copy_dcm(p, dicom_dataset, separated_output_dir)