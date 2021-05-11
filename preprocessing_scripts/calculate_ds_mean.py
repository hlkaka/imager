from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
import sys
import numpy as np
import tqdm
import torch

sys.path.append('data/')
sys.path.append('.')

from CTDataSet import CTDicomSlicesMaskless
from CustomTransforms import Window
from CustomTransforms import Imagify
from constants import Constants

def custom_collate_fn(batches):
    '''
    This function returns means and stds instead of the original Dataset batch
    This helps but simplifying the claculations on datasets with images of different sizes
    Use on Datasets with n_surrounding = 0
    '''
    means = []
    stds = []
    for tuple in batches:
        image = tuple[0]
        means.append(np.mean(image))
        stds.append(np.std(image))

    return torch.FloatTensor(means), torch.FloatTensor(stds)

if __name__ == '__main__':
    n_workers = input("Enter number of works (usually = num CPU cores) - no error checking:")
    n_workers = int(n_workers)

    dataset = Constants.ct_only_cleaned
    dcm_list = CTDicomSlicesMaskless.generate_file_list(dataset, dicom_glob='/*/*/dicoms/*.dcm')
    
    prep = transforms.Compose([Window(50, 250), Imagify(50, 250)])
    ctds = CTDicomSlicesMaskless(dcm_list, preprocessing=prep, n_surrounding=0)

    dl = DataLoader(ctds, batch_size=100, num_workers=n_workers, collate_fn=custom_collate_fn)

    means = 0
    stds = 0

    for m, s in tqdm.tqdm(dl):
        means = means + torch.sum(m)
        stds = stds + torch.sum(s)

    print('For dataset: {}\nMean: {}   -- STD: {}'.format(dataset, means/len(ctds), stds/len(ctds)))
