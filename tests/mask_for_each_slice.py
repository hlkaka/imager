'''
Ensures that the number of slices matches the number of mask images for every patient
'''

import glob
import os
import tqdm

organized_dataset = "/home/hussam/organized_dataset/"

# pt_id: dcm_slices, GTVps, GTVns, n slices == n all masks
data = {}

failed : bool = False

for p in tqdm.tqdm(glob.glob(organized_dataset + "*")):
    pt_id = os.path.basename(p)
    data[pt_id] = [0, 0, 0, 1]

    n_slices = len(glob.glob(p + "/dicoms/*.dcm"))
    data[pt_id][0] = n_slices

    GTVps = glob.glob(p + "/GTVp*")
    data[pt_id][1] = len(GTVps)

    for GTVp in GTVps:
        n_masks = len(glob.glob(GTVp + "/*.png"))
        
        if n_masks != n_slices:
            failed = True
            print("Pt: {} - n_slices: {} - GTVps: {}".format(pt_id, n_slices, GTVp))
            data[pt_id][3] = 0

    GTVns = glob.glob(p + "/GTVn*")
    data[pt_id][2] = len(GTVns)

    for GTVn in GTVns:
        n_masks = len(glob.glob(GTVn + "/*.png"))
        
        if n_masks != n_slices:
            failed = True
            print("Pt: {} - n_slices: {} - GTVns: {}".format(pt_id, n_slices, GTVn))
            data[pt_id][3] = 0

if failed:
    print("Errors occured. See above.")
else:
    print("Number of slices match number of masks for all patients. Success!")