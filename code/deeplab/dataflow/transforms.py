import numpy as np

import torch

from ignite.utils import convert_tensor

from albumentations import BasicTransform


class ToTensor(BasicTransform):
    
    keys = ['image', 'mask']

    def __call__(self, force_apply=False, **kwargs):
        
        for key in self.keys:
            if key not in kwargs:
                continue
            value = kwargs[key]
            if value.ndim == 3:
                value = value.transpose(2, 0, 1)
            kwargs[key] = torch.from_numpy(value)
        return kwargs


# class MaskOHE:

#     def __call__(self, **kwargs):
        
#         assert 'mask' in kwargs, "Input should contain 'mask'"
        
#         mask = kwargs['mask']
#         kwargs['mask'] = mask

#         return kwargs

def ignore_mask_boundaries(force_apply, **kwargs):
    assert 'mask' in kwargs, "Input should contain 'mask'"
    mask = kwargs['mask']
    mask[mask == 255] = 0
    kwargs['mask'] = mask
    return kwargs


def prepare_batch_fp32(batch, device, non_blocking):
    x, y = batch['image'], batch['mask']
    x = convert_tensor(x, device, non_blocking=non_blocking)
    y = convert_tensor(y, device, non_blocking=non_blocking).long()
    return x, y


def prepare_batch_fp16(batch, device, non_blocking):
    x, y = batch['image'], batch['mask']
    x = convert_tensor(x, device, non_blocking=non_blocking).half()
    y = convert_tensor(y, device, non_blocking=non_blocking).long()
    return x, y
