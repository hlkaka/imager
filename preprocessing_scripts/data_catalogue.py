'''
Generates a CSV file listing all patients and the number of slices in three planes
Uses nii files for this task
'''

import glob
import nibabel as nb
import numpy as np
import pandas as pd
import os

dataset = "/home/hussam/dataset/"
imgs_f = "image.nii.gz"
output_file = "/home/hussam/dataset/catalogue.csv"

pts = glob.glob(dataset + "*")
sizes = np.zeros((len(pts), 3))

for i, p in enumerate(pts):
    try:
        #print(i)
        im = nb.load(p + "/" + imgs_f)
        sizes[i,:] = im.shape
    except:
        print(p)

pts_basenames = map(os.path.basename, pts)

d = {'Patients': pts_basenames, 'X': sizes[:,0], 'Y':sizes[:, 1], 'Z':sizes[:,2]}
df = pd.DataFrame(data=d)

df.to_csv(output_file)