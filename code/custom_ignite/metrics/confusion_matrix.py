import numpy as np
from sklearn.metrics import confusion_matrix

import torch

from ignite.metrics import Metric, MetricsLambda
from ignite.exceptions import NotComputableError
from ignite.utils import to_onehot


class ConfusionMatrix(Metric):
    """Calculates confusion matrix for multiclass data.

    - `update` must receive output of the form `(y_pred, y)`.
    - `y_pred` must contain prediction logits and be in the following shape (batch_size, num_categories, ...)
    - `y` must contain ground-truth class indices and be in the following shape (batch_size, ...).

    Input data can be in format of images, e.g. `[B, C, H, W]` and `[B, H, W]`, and should contain background index equal 0.

    Args:
        num_classes (int): number of classes. In case of images, num_classes should also count the background index 0.
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
    """

    def __init__(self, num_classes, average_samples=False, output_transform=lambda x: x):
        self.num_classes = num_classes
        self._num_examples = 0
        self.average_samples = average_samples
        super(ConfusionMatrix, self).__init__(output_transform=output_transform)

    def reset(self):
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.float)
        self._num_examples = 0

    def _check_shape(self, output):
        y_pred, y = output

        if y.ndimension() > 1 and y.shape[1] == 1:
            # (N, 1, ...) -> (N, ...)
            y = y.squeeze(dim=1)

        if y_pred.ndimension() > 1 and y_pred.shape[1] == 1:
            # (N, 1, ...) -> (N, ...)
            y_pred = y_pred.squeeze(dim=1)

        if y.ndimension() + 1 != y_pred.ndimension():
            raise ValueError("y must have shape of (batch_size, ...) and y_pred must have "
                             "shape of (batch_size, num_categories, ...), "
                             "but given {} vs {}.".format(y.shape, y_pred.shape))

        y_shape = y.shape
        y_pred_shape = y_pred.shape

        if y.ndimension() + 1 == y_pred.ndimension():
            y_pred_shape = (y_pred_shape[0],) + y_pred_shape[2:]

        if y_shape != y_pred_shape:
            raise ValueError("y and y_pred must have compatible shapes.")

        # Too much computational cost for image input data
        # Let it fail at to_onehot
        # if y.max() > self.num_classes:
        #     raise ValueError("y max value is larger than the number of classes")

        return y_pred, y

    def update(self, output):
        y_pred, y = self._check_shape(output)

        _, indices = torch.max(y_pred, dim=1)
        y_pred_ohe = to_onehot(indices.reshape(-1), self.num_classes)
        y_ohe = to_onehot(y.reshape(-1), self.num_classes)
        
        y_ohe_t = y_ohe.transpose(0, 1).float()        
        y_pred_ohe = y_pred_ohe.float()

        if self.confusion_matrix.type() != y_ohe_t.type():
            self.confusion_matrix = self.confusion_matrix.type_as(y_ohe_t)

        self.confusion_matrix += (y_ohe_t @ y_pred_ohe).float()
        self._num_examples += y_pred.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Confusion matrix must have at least one example before it can be computed.')
        if self.average_samples:
            return self.confusion_matrix / self._num_examples
        return self.confusion_matrix.cpu()


def IoU(cm, ignore_background=True):
    assert isinstance(cm, ConfusionMatrix)    
    iou = cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) - cm.diag())
    if ignore_background:
        return MetricsLambda(lambda res: res[1:], iou)
    else: 
        return iou


def mIoU(cm, ignore_background=True):
    return IoU(cm=cm, ignore_background=ignore_background).mean()


def Accuracy(cm):
    return cm.diag().sum() / cm.sum()


def Precision(cm):
    return (cm.diag() / cm.sum(dim=0)).mean()


def Recall(cm):
    return (cm.diag() / cm.sum(dim=1)).mean()
