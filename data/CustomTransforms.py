import numpy as np
from albumentations import ToFloat, FromFloat
from torchvision import transforms
import torch

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
    def GaussianNoise(tensor, mean = 0, std = 1):
        '''
        Inplace
        Works on single images and on batches
        '''
        tensor += torch.randn(tensor.size(), dtype=tensor.dtype) * std + mean
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