import os
from itertools import chain
from functools import partial

import cv2
import torch
import torch.nn as nn
import torch.optim as optim

from albumentations import Compose, RandomCrop, ShiftScaleRotate, HorizontalFlip, PadIfNeeded, CenterCrop
from albumentations import GaussNoise, RandomBrightnessContrast, Normalize

from dataflow.dataloaders import get_train_val_loaders
from dataflow.datasets import get_train_dataset
from dataflow.transforms import ToTensor, ignore_mask_boundaries, prepare_batch_fp16, prepare_batch_fp32

from models.deeplabv3 import DeepLabV3
from models.backbones import build_resnet18_backbone

assert 'DATASET_PATH' in os.environ
data_path = os.environ['DATASET_PATH']


debug = True
use_time_profiling = True


use_time_profiling = True

seed = 12
device = 'cuda'

padded_img_size = 600
img_size = 513

train_transforms = Compose([
    HorizontalFlip(),
    PadIfNeeded(padded_img_size, padded_img_size, border_mode=cv2.BORDER_CONSTANT),
    ShiftScaleRotate(shift_limit=(-0.05, 0.05), 
                     scale_limit=(-0.1, 2.0), 
                     rotate_limit=45, 
                     p=0.8, 
                     border_mode=cv2.BORDER_CONSTANT),
    RandomCrop(img_size, img_size),

    GaussNoise(p=0.5),
    RandomBrightnessContrast(),
    
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ignore_mask_boundaries,
    ToTensor(),
])
train_transform_fn = lambda dp: train_transforms(**dp)


val_transforms = Compose([
    PadIfNeeded(padded_img_size, padded_img_size, border_mode=cv2.BORDER_CONSTANT),    
    CenterCrop(img_size, img_size),

    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ignore_mask_boundaries,
    ToTensor(),    
])
val_transform_fn = lambda dp: val_transforms(**dp)


batch_size = 8
num_workers = 10
non_blocking = True


train_loader, val_loader, train_eval_loader = get_train_val_loaders(root_path=data_path, 
                                                                    train_transforms=train_transform_fn,
                                                                    val_transforms=val_transform_fn,
                                                                    batch_size=batch_size,
                                                                    num_workers=num_workers,
                                                                    val_batch_size=batch_size * 2,
                                                                    limit_train_num_samples=250 if debug else None,
                                                                    limit_val_num_samples=250 if debug else None,                                                             
                                                                    random_seed=seed)

prepare_batch = prepare_batch_fp32

val_interval = 5

num_classes = 21
model = DeepLabV3(build_resnet18_backbone, num_classes=num_classes)


criterion = nn.CrossEntropyLoss()

lr = 0.007 / 4.0 * batch_size
weight_decay = 5e-4
momentum = 0.9


optimizer = optim.SGD([{'params': model.backbone.parameters()}, 
                       {'params': chain(model.aspp.parameters(), model.decoder.parameters())}],
                      lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)

num_epochs = 50 if not debug else 2


l = len(train_loader)

def lambda_lr_scheduler(iteration, lr0, n, a):
    return lr0 * pow((1.0 - 1.0 * iteration / n), a)


lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, 
                                           lr_lambda=[
                                               partial(lambda_lr_scheduler, lr0=lr, n=num_epochs * l, a=0.9),
                                               partial(lambda_lr_scheduler, lr0=lr * 10.0, n=num_epochs * l, a=0.9)
                                           ])

def score_function(evaluator):
    score = evaluator.state.metrics['mIoU']
    return score

