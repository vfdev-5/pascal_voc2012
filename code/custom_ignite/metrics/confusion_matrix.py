import numpy as np
from sklearn.metrics import confusion_matrix

import torch

from ignite.metrics import Metric, MetricsLambda
from ignite.exceptions import NotComputableError
from ignite.utils import to_onehot


def output_gt_predicted_classes(output):
    y_logits, y = output            
    # We assume y_logits to be composed of logits [pr1, pr2, pr3, ...] at dim=1
    # without background => compute argmax
    _, y_pred = torch.max(y_logits, dim=1)
    return y_pred, y


def output_gt_predicted_classes_bg(output):
    y_logits, y = output            
    # We assume y_pred to be composed of probabilities [pr1, pr2, pr3, ...] at dim=1
    # without background => compute class probas -> background probas -> argmax
    y_probas = torch.sigmoid(y_logits)
    bg_probas = (1.0 - y_probas.sum(dim=1)).unsqueeze(dim=1)
    y_probas_w_bg = torch.cat([bg_probas, y_probas], dim=1)
    _, y_pred = torch.max(y_probas_w_bg, dim=1)
    return y_pred, y


class ConfusionMatrix_slow(Metric):
    """Calculates confusion matrix for multiclass data.

    - `update` must receive output of the form `(y_pred, y)`.
    - `y_pred` must contain predicted class indices and be in the following shape (batch_size, ...)
    - `y` must contain ground-truth class indices and be in the following shape (batch_size, ...).

    Input data can be in format of images, e.g. `[B, H, W]`, and should contain background index equal 0.

    User can compute predicted class indices from logits using `output_transform`:

    .. code-block:: python

        def output_gt_predicted_classes(output):
            y_logits, y = output            
            # We assume y_pred to be composed of probabilities [pr1, pr2, pr3, ...] at dim=1
            
            # a) compute indices with background
            y_probas = torch.sigmoid(y_logits)
            bg_probas = (1.0 - y_probas.sum(dim=1)).unsqueeze(dim=1)
            y_probas_w_bg = torch.cat([bg_probas, y_probas], dim=1)
            _, y_pred = torch.max(y_probas_w_bg, dim=1)

            # b) compute indices without background
            # y_pred = torch.max(y_logits, dim=1)
            
            return y_pred, y

        cm = ConfusionMatrix(num_classes, output_gt_predicted_classes)


    Args:
        num_classes (int): number of classes. In case of images, num_classes should also count the background index 0.
        output_transform (callable, optional): a callable that is used to transform the
            :class:`~ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric. This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
    """

    def __init__(self, num_classes, output_transform=lambda x: x):
        self.num_classes = num_classes
        self._num_examples = 0
        self._labels = np.arange(num_classes)
        super(ConfusionMatrix_slow, self).__init__(output_transform=output_transform)

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

        if y.ndimension() != y_pred.ndimension():
            raise ValueError("y must have shape of (batch_size, ...) and y_pred must have "
                             "shape of (batch_size, ...), "
                             "but given {} vs {}.".format(y.shape, y_pred.shape))

        y_shape = y.shape
        y_pred_shape = y_pred.shape

        if y_shape != y_pred_shape:
            raise ValueError("y and y_pred must have compatible shapes.")

        return y_pred, y

    def update(self, output):
        y_pred, y = self._check_shape(output)

        batch_size = y_pred.shape[0]
        np_indices = y_pred.reshape(batch_size, -1).cpu().numpy()
        np_y = y.reshape(batch_size, -1).cpu().numpy()

        for i in range(batch_size):
            cm = confusion_matrix(np_y[i, :], np_indices[i, :], labels=self._labels)
            self.confusion_matrix += torch.from_numpy(cm).type_as(self.confusion_matrix)
            self._num_examples += 1

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Confusion matrix must have at least one example before it can be computed.')
        return self.confusion_matrix / self._num_examples


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

        if y.max() > self.num_classes:
            raise ValueError("y max value is larger than the number of classes")

        return y_pred, y

    def update(self, output):
        y_pred, y = self._check_shape(output)

        _, indices = torch.max(y_pred, dim=1)
        y_pred_ohe = to_onehot(indices.reshape(-1), self.num_classes)
        y_ohe = to_onehot(y.reshape(-1), self.num_classes)
        
        for a in range(self.num_classes):
            p = y_ohe[:, a:a+1] * y_pred_ohe
            for b in range(self.num_classes):
                self.confusion_matrix[a, b] += p[:, b].sum()                
        self._num_examples += y_pred.shape[0]

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('Confusion matrix must have at least one example before it can be computed.')
        if self.average_samples:
            return self.confusion_matrix / self._num_examples
        return self.confusion_matrix



def IoU(ignore_background=True, *args, **kwargs):
    cm = ConfusionMatrix(*args, **kwargs)
    iou = cm.diag() / (cm.sum(dim=1) + cm.sum(dim=0) - cm.diag())
    if ignore_background:
        return MetricsLambda(lambda res: res[1:], iou)
    else: 
        return iou


def mIoU(ignore_background=True, *args, **kwargs):
    return IoU(ignore_background=ignore_background, *args, **kwargs).mean()
