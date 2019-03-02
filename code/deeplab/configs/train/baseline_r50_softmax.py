
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim

from albumentations import Compose, Resize, ShiftScaleRotate, GaussNoise, Ran
from albumentations import RandomBrightnessContrast, Normalize

from ignite.contrib.handlers import PiecewiseLinear

from dataflow.dataloaders import get_train_val_loaders
from dataflow.datasets import get_train_dataset
from dataflow.transforms import ToTensor, ignore_mask_boundaries, prepare_batch_fp16, prepare_batch_fp32

from models.deeplabv3 import DeepLabV3
from models.backbones import build_resnet50_backbone

assert 'DATASET_PATH' in os.environ
data_path = os.environ['DATASET_PATH']

seed = 12
device = 'cuda'

use_fp16 = False


train_transforms = Compose([
    ShiftScaleRotate(shift_limit=0.2, scale_limit=0.075, rotate_limit=45, interpolation=cv2.INTER_CUBIC, p=0.3),
    Resize(224, 224, interpolation=cv2.INTER_CUBIC),
    GaussNoise(),
    RandomBrightnessContrast(),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ignore_mask_boundaries,
    ToTensor(),
])
train_transform_fn = lambda dp: train_transforms(**dp)


val_transforms = Compose([
    Resize(224, 224, interpolation=cv2.INTER_CUBIC),
    Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ignore_mask_boundaries,
    ToTensor(),
])
val_transform_fn = lambda dp: val_transforms(**dp)


batch_size = 64

non_blocking = True


train_loader, val_loader, train_eval_loader = get_train_val_loaders(root_path=data_path, 
                                                                    train_transforms=train_transform_fn,
                                                                    val_transforms=val_transform_fn,
                                                                    batch_size=batch_size,
                                                                    num_workers=10,
                                                                    val_batch_size=batch_size * 2,                                                                    
                                                                    random_seed=seed)

prepare_batch = prepare_batch_fp16 if use_fp16 else prepare_batch_fp32

val_interval = 5

num_classes = 21
model = DeepLabV3(build_resnet50_backbone, num_classes=num_classes)


criterion = nn.CrossEntropyLoss()

lr = 0.007 * batch_size
weight_decay = 1e-4

optimizer = optim.SGD(model.parameters(), 
                      lr=lr / batch_size,                       
                      weight_decay=weight_decay * batch_size)

num_epochs = 50


l = len(train_loader)
lr_scheduler = PiecewiseLinear(optimizer, 
                               param_name="lr",
                               milestones_values=[(0, 0), (5 * l, lr), (35 * l, lr), (35 * l, lr * 0.1), (num_epochs * l, 0.0)])


def score_function(evaluator):
    score = evaluator.state.metrics['mIoU']
    return score

