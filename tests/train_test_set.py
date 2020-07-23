import os
import random
import sys
sys.path.append('data/')

from CTDataSet import CTDicomSlices, DatasetManager

dataset = "../organized_dataset"

if __name__ == '__main__':
    #dsm = DatasetManager.generate_train_val_test(dataset)

    dsm = DatasetManager.load_train_val_test(dataset, "train.txt", "val.txt", "test.txt")

    train_p, val_p, test_p = dsm.train, dsm.val, dsm.test
    train, val, test = dsm.get_dicoms()

    lists = {"TRAIN": (train, train_p), "VAL": (val, val_p), "TEST": (test, test_p)}

    for k in lists:
        print("{} SET\n{} patients with {} .dcm files\nRandom pt: {}\nRandom .dcm: {}\n\n".format(
            k, len(lists[k][1]), len(lists[k][0]), random.choice(lists[k][1]), random.choice(lists[k][0])
        ))
    
    #dsm.save_lists(".")