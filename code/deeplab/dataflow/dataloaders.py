
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from dataflow.datasets import get_train_dataset, get_val_dataset, TransformedDataset


def get_train_val_loaders(root_path, train_transforms, val_transforms, 
                          batch_size=16, num_workers=8, val_batch_size=None,
                          pin_memory=True,
                          random_seed=None, 
                          limit_train_num_samples=None, 
                          limit_val_num_samples=None):
    
    train_ds = get_train_dataset(root_path)
    val_ds = get_val_dataset(root_path)

    if random_seed is not None:
        np.random.seed(random_seed)

    if limit_train_num_samples is not None:
        train_indices = np.random.permutation(limit_train_num_samples)
        train_ds = Subset(train_ds, train_indices)
    
    if limit_val_num_samples is not None:
        val_indices = np.random.permutation(limit_val_num_samples)
        val_ds = Subset(val_ds, val_indices)

    # random samples for evaluation on training dataset
    if len(val_ds) * 10 < len(train_ds):
        train_eval_indices = np.random.permutation(len(val_ds))
        train_eval_ds = Subset(train_ds, train_eval_indices)
    else:
        train_eval_ds = train_ds

    train_ds = TransformedDataset(train_ds, transform_fn=train_transforms)
    val_ds = TransformedDataset(val_ds, transform_fn=val_transforms)
    train_eval_ds = TransformedDataset(train_eval_ds, transform_fn=val_transforms)

    train_loader = DataLoader(train_ds, shuffle=True,
                              batch_size=batch_size, num_workers=num_workers,
                              pin_memory=pin_memory, drop_last=True)

    val_batch_size = batch_size * 4 if val_batch_size is None else val_batch_size
    val_loader = DataLoader(val_ds, shuffle=False,
                            batch_size=val_batch_size, num_workers=num_workers,
                            pin_memory=pin_memory, drop_last=False)

    train_eval_loader = DataLoader(train_eval_ds, shuffle=False,
                                   batch_size=val_batch_size, num_workers=num_workers,
                                   pin_memory=pin_memory, drop_last=False)

    return train_loader, val_loader, train_eval_loader
