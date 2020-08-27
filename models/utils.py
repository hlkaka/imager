'''
Common data opertations, such as data agumentations, which are
expected to be used on multiple models
'''

import torch
from CustomTransforms import TorchFunctionalTransforms as TFT
from torch.cuda.amp import autocast

class Utils:
    @staticmethod
    def preprocessing(images, masks, WL, WW, b_h_w_c = True):
        '''
        Applies windowing to input
        If input is (B, H, W, C), then set the b_h_w_c to true so the model
        can convert the image to (B, C, H, W)
        '''
        if b_h_w_c:
            images = images.permute(0, 3, 1, 2)

        with torch.no_grad():
            TFT.Window(images, WL, WW)
            TFT.Imagify(images, WL, WW)

        return images, masks

    @staticmethod
    def do_train_augmentations(images, masks, gaussian_noise_std, device, ra, rf):
        '''
        NOT inplace
        '''
        # Kornia affine transforms do torch.inverse which is not yet implemented for FP16
        # torch.cuda.amp does not yet blacklist torch.inverse in autocast
        # therefore, need to explicitly disable autocast to force float32 for this operation
        # this should, in theory, have no effect if percision = 16 is not set
        with autocast(enabled=False):
            with torch.no_grad():
                TFT.GaussianNoise(images, std = gaussian_noise_std, device=device)
            
                # add the equivalent of a channel axis to masks so Kornia can work with it
                # We expect (B, C, H, W) so the channel axis goes in position 1
                masks = masks.unsqueeze(1)

                params = ra.generate_parameters(images.shape)
                images = ra(images, params)
                masks = ra(masks, params)

                params = rf.generate_parameters(images.shape)
                images = rf(images, params)
                masks = rf(masks, params)

                # Remove the mask channel axis
                masks = masks.squeeze(1)

        return images, masks