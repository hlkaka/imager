'''
Some .dcm files will have skipped indices due to skipped slice
numbers in the DICOM header.
This file ensures sequential numbering of slices starting with 1.dcm
This should make fix_offset_numbering obsolete
'''

import tqdm
import glob
import os

dataset = '../organized_dataset'

fixed_pts = []

for p in tqdm.tqdm(glob.glob("{}/*".format(dataset))):
    dcms = glob.glob("{}/dicoms/*.dcm".format(p))
    
    get_slice_n = lambda f: int(os.path.basename(f)[0:-4])
    slice_n = list(map(get_slice_n, dcms))
    slice_n.sort()

    min_sl, max_sl = min(slice_n), max(slice_n)
    if min_sl != 1 or max_sl != len(slice_n):
        fixed_pts.append([os.path.basename(p), min_sl, max_sl, len(slice_n)])

        # slice_n must be sorted at this point
        for i, sl in enumerate(slice_n):
            if i + 1 != sl:   # add 1 to i for 1-based indexing
                 os.rename("{}/dicoms/{}.dcm".format(p, sl), "{}/dicoms/{}.dcm".format(p, i + 1))

print("Done.")
print("Processed {} patients as follows:".format(len(fixed_pts)))

for fp in fixed_pts:
    print("{} w/ {} slices: {}-{}  -->  1-{}".format(fp[0], fp[3], fp[1], fp[2], fp[3]))

# Output
# Done.
# Processed 5 patients as follows:
# HNSCC-01-0415 w/ 239 slices: 1-242  -->  1-239
# HNSCC-01-0502 w/ 266 slices: 1-267  -->  1-266
# HNSCC-01-0513 w/ 267 slices: 1-268  -->  1-267
# HNSCC-01-0464 w/ 132 slices: 1-160  -->  1-132
# HNSCC-01-0352 w/ 287 slices: 1-288  -->  1-287