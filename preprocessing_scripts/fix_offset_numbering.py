'''
Fixes the offset for given patients.
E.g. if a pt's dicoms are 5.dcm to 15.dcm, this would change it to 1.dcm to 11.dcm
Note: this is obsolete. Use fix_dcm_numbering instead
'''

import tqdm
import glob
import os

dataset = '../organized_dataset/'
pts = ['HNSCC-01-0337', 'HNSCC-01-0309', 'HNSCC-01-0517', 'HNSCC-01-0607', 'HNSCC-01-0352']

for p in tqdm.tqdm(pts):
    dcms = glob.glob(dataset + p + "/dicoms/*.dcm")
    
    get_slice_n = lambda f: int(os.path.basename(f)[0:-4])
    slice_n = list(map(get_slice_n, dcms))
    slice_n.sort()

    offset = slice_n[0] - 1

    print("Patient {} has offset of {}".format(p, offset))

    # At this point, list is sorted
    for sl in slice_n:
        os.rename("{}{}/dicoms/{}.dcm".format(dataset, p, sl), "{}{}/dicoms/{}.dcm".format(dataset, p, sl - offset))

# Run results
#Patient HNSCC-01-0337 has offset of 12
#Patient HNSCC-01-0309 has offset of 27
#Patient HNSCC-01-0517 has offset of 19
#Patient HNSCC-01-0607 has offset of 65
#Patient HNSCC-01-0352 has offset of 1