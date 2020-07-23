'''
Create and loads a holdout set
'''

import glob
import os
import random
import shutil

holdout_frac = 0.1
dataset = '../organized_dataset'
holdout_list = '../holdout_set.txt'
holdout_dir = '../holdouts'

# This decides where to input a list of holdouts or to create a new list
create_holdout = False

def read_list(holdout_list):
    txt = None

    with open(holdout_list, "r") as f:
        txt = f.read()

    return [x.strip() for x in txt.split('\n')]

def write_list(holdout_list, hodlout_pts):
    txt = '\n'.join(hodlout_pts)

    with open(holdout_list, "w") as f:
        f.write(txt)

def do_holdout(dataset :str, holdout_list :str, holdout_dir :str, create_holdout :bool, holdout_frac :float = 0.1):
    '''
    Holds out patients.
    If create_holdout, then creates using the given fraction.
    Otherwise, uses the given list.
    Holdout_list is the path to a text file containing the list of the holdout patients
    '''
    ds_pts = glob.glob("{}/*".format(dataset))
    all_pts = [os.path.basename(x) for x in ds_pts]

    if create_holdout:
        hodlout_pts = random.sample(all_pts, int(holdout_frac * len(all_pts)))
    else:
        hodlout_pts = read_list(holdout_list)

    os.makedirs(holdout_dir, exist_ok=True)

    p_moved = []
    for p in hodlout_pts:
        if os.path.isdir("{}/{}".format(dataset, p)):
            shutil.move("{}/{}".format(dataset, p), "{}/{}".format(holdout_dir, p))
            p_moved.append(p)

    write_list(holdout_list, p_moved)

    # print results
    print("Moved {} patients to holdout set in {}".format(len(p_moved), holdout_dir))
    print("The following patients were moved: {}".format(', '.join(p_moved)))

if __name__ == '__main__':
    do_holdout(dataset, holdout_list, holdout_dir, create_holdout, holdout_frac=holdout_frac)