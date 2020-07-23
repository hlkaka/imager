'''
Finds patients for whom the first DICOM slice does not start with 1
'''

import glob
import os
import tqdm

dataset = '../organized_dataset/'

pt_folders = glob.glob(dataset + "*")
offset_pts = []

for pt_f in tqdm.tqdm(pt_folders):
    if not os.path.isfile(pt_f + "/dicoms/1.dcm"):
        offset_pts.append(pt_f)

print("Stats:")
print("# patients: {}".format(len(pt_folders)))
print("# patients without 1.dcm: {}  --  {:.2f}%".format(len(offset_pts), len(offset_pts) / len(pt_folders) * 100))
print("Pts missing 1.dcm:")

for pt_f in offset_pts:
    print(os.path.basename(pt_f))