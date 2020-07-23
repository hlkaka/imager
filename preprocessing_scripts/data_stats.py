import glob
import numpy as np
import tqdm

import sys
sys.path.append('data/')

from CTDataSet import CTDicomSlices

def print_stats(dataset, n_slices, pos_masks, pos_pixels, total_pixels):
    print("Location: {}".format(dataset))
    print("# patients: {}".format(dataset))
    print("# slices: {}".format(n_slices))
    print("% positive masks: {:.2f}%  --  # positive masks: {}".format(positive_masks / n_slices * 100, positive_masks))
    print("% positive pixels: {:.2f}%  -- # positive pixels: {}".format(positive_pixels / total_pixels * 100, positive_pixels))

if __name__ == '__main__':
    dataset = '/home/hussam/organized_dataset/'
    dcm_list = CTDicomSlices.generate_file_list(dataset)
    ctds = CTDicomSlices(dcm_list)

    print("Dataset is loaded.")

    n_pts = len(glob.glob(dataset + "*"))
    n_slices = len(ctds)
    
    image_size = 512 * 512

    positive_masks = 0
    positive_pixels = 0

    i = 0

    for _, mask, _, _ in tqdm.tqdm(ctds):
        pixels = np.sum(mask)

        positive_pixels += int(pixels)

        if pixels > 0:
            positive_masks += 1

        i = i + 1
        if i != 0 and i % 10000 == 0:
            print("Interim Stats at {}".format(i))
            print_stats(dataset, i, positive_masks, positive_pixels, image_size * i)

    print("Final Stats:")
    print_stats(dataset, n_slices, positive_masks, positive_pixels, image_size * n_slices)

# Output
# Final Stats:
# Location: /home/hussam/organized_dataset/
# # patients: /home/hussam/organized_dataset/
# # slices: 101363
# % positive masks: 19.37%  --  # positive masks: 19633
# % positive pixels: 0.15%  -- # positive pixels: 39514776

# % positive pixels with more precision: 14.87099907845904 %