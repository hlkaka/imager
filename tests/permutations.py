import sys
sys.path.append('.')

from constants import Constants
from data.CTDataSet import CTDicomSlicesJigsaw

if __name__ == '__main__':
    dataset = Constants.ct_only_cleaned
    dcm_list = CTDicomSlicesJigsaw.generate_file_list(dataset,
        dicom_glob='/*/*/dicoms/*.dcm')
    ctds = CTDicomSlicesJigsaw(dcm_list, trim_edges=True,
            return_tile_coords=True, normalize_tiles=False, max_pixel_value=255)
        
    print(ctds.generate_permutations(10))