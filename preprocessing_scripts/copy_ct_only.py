'''
This script copies only CT images for the given dataset to the target
directory. It preserves the directory structure for directories which
contain CT series, but does not copy any directory without a CT series.
'''

import os
import pydicom
import shutil

#source_dataset = "E:\\HNSCC dataset\\qin\\QIN-HEADNECK"   # Make sure this dir has a dir for each pt
#dest_dataset = "G:\\thesis\\ct_only\\qin"

dir_sep = "\\"    # For Unix, use "/"

min_n_slices = 10 # Do not use any series with less than 10 slices.
                  # This will eliminate RT struct series

def process_directory(source_dir_base :str, dest_dir_base :str, file_list :str):
    '''
    Recursively processes the given directory using os.walk
    '''
    dir_counter = 1

    for path, _, fileList in os.walk(source_dir_base):
        if len(fileList) > min_n_slices:
            # Verify that modality is CT
            with pydicom.dcmread(path + dir_sep + fileList[0]) as ds:
                modality = ds[0x0008, 0x0060].value
            
            suffix = path[len(source_dir_base) + 1:]     # +1 to remove the dir separator for consistency

            if modality == 'CT':
                dest = "{}{}{}".format(dest_dir_base, dir_sep, dir_counter) 
                os.makedirs(dest, exist_ok=True)

                file_list_str = ""
                for f in fileList:
                    shutil.copy2("{}{}{}".format(path, dir_sep, f), "{}{}{}".format(dest, dir_sep, f))
                    file_list_str += "{},{}\n".format("{}{}{}".format(suffix, dir_sep, f), "{}{}{}".format(dir_counter, dir_sep, f))

                with open(dest_dir_base + dir_sep + file_list, "a+") as file_list_file:
                    file_list_file.write(file_list_str)
                
                dir_counter += 1

if __name__ == '__main__':
    datasets = {
        "E:\\HNSCC dataset\\qin": "G:\\thesis\\ct_only\\qin",
        "E:\\HNSCC dataset\\OPC-radiomics": "G:\\thesis\\ct_only\\OPC-radiomics",
        "E:\\HNSCC dataset\\HNSCC 2": "G:\\thesis\\ct_only\\HNSCC 2",
        "G:\\thesis\\cetuximab": "G:\\thesis\\ct_only\\cetuximab",
        "G:\\thesis\\head-neck-pet-ct": "G:\\thesis\\ct_only\\head-neck-pet-ct",
        "G:\\thesis\\head-neck-radiomics": "G:\\thesis\\ct_only\\head-neck-radiomics",
        "G:\\thesis\\head-neck-scc": "G:\\thesis\\ct_only\\head-neck-scc",
        "G:\\thesis\\tcga": "G:\\thesis\\ct_only\\tcga",
    }
    
    for src in datasets:
        print("Processing {} to {}".format(src, datasets[src]))
        process_directory(src, datasets[src], "dcm_list.txt")

