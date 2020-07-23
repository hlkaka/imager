'''
Finds the patient IDs which exist in original_ds but not target_ds
Usually this is due to a missing RT-STRUCT DICOM file
'''

import glob
import os

original_ds = "/mnt/e/HNSCC dataset/HNSCC"
target_ds = "/home/hussam/dataset"

for f in glob.glob(original_ds + "/*"):
    base_name = os.path.basename(f)

    if not os.path.isdir(target_ds + "/" + base_name):
        print(base_name)