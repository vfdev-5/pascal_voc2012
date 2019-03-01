import numpy as np

import cv2

from PIL import Image


from torch.utils.data import Dataset
from torchvision.datasets.voc import VOCSegmentation


class TransformedDataset(Dataset):

    def __init__(self, ds, transform_fn):
        assert isinstance(ds, Dataset)
        assert callable(transform_fn)
        self.ds = ds
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        dp = self.ds[index]
        return self.transform_fn(dp)


def datapoint_to_dict(dp):
    return {
        "image": dp[0],
        "mask": dp[1]
    }


class VOCSegmentationOpencv(VOCSegmentation):

    def __getitem__(self, index):
        img = cv2.imread(self.images[index])
        assert img is not None, "Image at '{}' has a problem".format(self.images[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = np.asarray(Image.open(self.masks[index]))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return {"image": img, "mask": mask}


def get_train_dataset(root_path):
    return VOCSegmentationOpencv(root=root_path, year='2012', image_set='train', download=False)


def get_val_dataset(root_path):    
    return VOCSegmentationOpencv(root=root_path, year='2012', image_set='val', download=False)                            
