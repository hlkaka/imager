import numpy as np
import torch
from skimage.transform import rescale

class TorchFunctionalTransforms():
    @staticmethod
    def Window(tensor, level, width):
        '''
        Inplace
        Works on single images and on batches
        '''
        Max = level + width // 2
        Min = level - width // 2
        torch.clamp(tensor, Min, Max, out=tensor)

        return tensor

    @staticmethod
    def Imagify(tensor, level=1024, width=4096, min_pix = 0, max_pix = 255):
        '''
        Inplace
        Works on single images and on batches
        '''
        min_hu = level - width // 2
        max_hu = level + width // 2

        # Tensor operations are in_place
        tensor -= min_hu
        tensor /= (max_hu - min_hu)
        tensor *= max_pix
        tensor += min_pix

        return tensor

    @staticmethod
    def GaussianNoise(tensor, mean = 0, std = 1, device = torch.device("cpu")):
        '''
        Inplace
        Works on single images and on batches
        '''
        tensor += torch.randn(tensor.size(), dtype=tensor.dtype, device=device) * std + mean
        return tensor

class Window():
    def __init__(self, level, width):
        '''
        Clips the given image using the given HU window level and width.
        '''
        self.max = level + width // 2
        self.min = level - width // 2

    def __call__(self, img):
        return np.clip(img, self.min, self.max)

class Imagify():
    def __init__(self, level = 1024, width = 4096, dtype = "uint8", min_pix = 0, max_pix = 255):
        '''
        Converts the given image from HU to the given display scale.
        Default is uint8 0 to 255.
        Default window is -1024 to +3076 (i.e. all physiologic HU values)
        '''        
        self.min_hu = level - width // 2
        self.max_hu = level + width // 2

        self.min_pix = min_pix
        self.max_pix = max_pix

        self.dtype = dtype

    def __call__(self, img):
        img = img - self.min_hu
        
        img = img.astype('float32')
        img = img / (self.max_hu - self.min_hu) * self.max_pix + self.min_pix

        return img.astype(self.dtype)

class Normalize():
    '''
    Normalizes the images using the given mean and std.
    For the pretraining set, mean 61.02492904663086   -- STD: 78.31950378417969
    '''
    def __init__(self, mean = 61.0249, std = 78.3195):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        return (img - self.mean) / self.std

class MinDimension():
    '''
    Resizes an image such that the smallest dimension is equal to the assgined value.
    Maintains aspect ratio.
    Assumes image shape is [H, W, C]
    '''
    def __init__(self, min_dimension):
        self.min_dimension = min_dimension

    def __call__(self, image):
        img_min_dim = np.min(image.shape[0:2])
        if self.min_dimension <= img_min_dim:
            return {'image': image}

        ratio = self.min_dimension / img_min_dim
        # order = 3 refers to bicubic
        new_img = rescale(image, ratio, order=3, channel_axis=-1, preserve_range=True)

        return {'image': new_img}