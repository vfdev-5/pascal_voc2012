import random

import numpy as np

import torch

import ignite
from ignite.engine import Engine, create_supervised_evaluator, Events
from ignite.metrics import RunningAverage, Precision, Recall, Accuracy
from ignite.handlers import EarlyStopping, ModelCheckpoint, TerminateOnNan
from ignite.contrib.handlers import ProgressBar

from polyaxon_client.tracking import get_outputs_path

from custom_ignite.contrib.config_runner.utils import set_seed

from custom_ignite.metrics import IoU, mIoU
from custom_ignite.metrics.confusion_matrix import output_gt_predicted_classes

from custom_ignite.contrib.handlers import TensorboardLogger
from custom_ignite.contrib.handlers.tensorboard_logger import output_handler as tb_output_handler, \
    optimizer_params_handler as tb_optimizer_params_handler, weights_scalar_handler as tb_weights_scalar_handler, \
    grads_scalar_handler as tb_grads_scalar_handler

from custom_ignite.contrib.handlers import PolyaxonLogger 
from custom_ignite.contrib.handlers.polyaxon_logger import output_handler as plx_output_handler


from utils import predictions_gt_images_handler


def run(config, logger):
    
    plx_logger = PolyaxonLogger()

    set_seed(config.seed)

    plx_logger.log_params(**{
        "seed": config.seed,
        "fp16": config.use_fp16,

        "pytorch version": torch.__version__,
        "ignite version": ignite.__version__,
        "cuda version": torch.version.cuda
    })

    device = config.device
    use_fp16 = config.use_fp16

    if use_fp16:
        assert "cuda" in device

    model = config.model.to(device)
    if use_fp16:
        model = model.half()

    optimizer = config.optimizer
    criterion = config.criterion.to(device)

    prepare_batch = config.prepare_batch
    non_blocking = config.non_blocking

    def train_update_function(engine, batch):

        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        return {
            'total_loss': loss.item()
        }

    trainer = Engine(train_update_function)

    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
    # Checkpoint training
    path = get_outputs_path()
    if path is None:
        path = "output"
    checkpoint_handler = ModelCheckpoint(dirname=path, 
                                         filename_prefix="checkpoint",
                                         save_interval=500)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, 
                              checkpoint_handler, 
                              {'model': model, 'optimizer': optimizer})

    RunningAverage(output_transform=lambda x: x['total_loss']).attach(trainer, 'total_loss')
    ProgressBar(persist=True).attach(trainer, ['total_loss'])

    def output_transform(output):
        return output['y_pred'], output['y']

    num_classes = config.num_classes
    val_metrics = {
        "mAcc": Accuracy(output_transform=output_transform), 
        "mPr": Precision(average=True, output_transform=output_transform),
        "mRe": Recall(average=True, output_transform=output_transform),
        # "IoU": IoU(num_classes=num_classes, output_transform=output_gt_predicted_classes),
        "mIoU": mIoU(num_classes=num_classes, 
                     output_transform=lambda o: output_gt_predicted_classes(output_transform(o))),
    }

    def eval_update_function(engine, batch):
        model.eval()
        with torch.no_grad():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred = model(x)
            return {
                "x": x, 
                "y_pred": y_pred, 
                "y": y
            }

    train_evaluator = Engine(eval_update_function)
    evaluator = Engine(eval_update_function)

    for name, metric in val_metrics.items():
        metric.attach(evaluator, name)
        metric.attach(train_evaluator, name)
    
    ProgressBar(persist=True, desc="Train Evaluation").attach(train_evaluator)
    ProgressBar(persist=True, desc="Val Evaluation").attach(evaluator)

    log_dir = get_outputs_path()
    if log_dir is None:
        log_dir = "debug_tb_logs"

    tb_logger = TensorboardLogger(log_dir=log_dir)

    # Log model's graph    
    # x, _ = prepare_batch(next(iter(config.train_loader)), device=device, non_blocking=True)
    # tb_logger.log_graph(model, x)

    tb_logger.attach(trainer,
                     log_handler=tb_output_handler(tag="training", output_transform=lambda x: x),
                     event_name=Events.ITERATION_COMPLETED)

    plx_logger.attach(trainer,
                      log_handler=plx_output_handler(tag="training", output_transform=lambda x: x),
                      event_name=Events.ITERATION_COMPLETED)

    if hasattr(config, "lr_scheduler"):
        trainer.add_event_handler(Events.ITERATION_STARTED, config.lr_scheduler)

    # @trainer.on(Events.STARTED)
    # def warmup_cudnn(engine):
    #     logger.info("Warmup CuDNN on random inputs")
    #     for _ in range(5):
    #         for size in [batch_size, len(test_loader.dataset) % batch_size]:
    #             warmup_cudnn(model, criterion, size, config)

    val_interval = config.val_interval
    @trainer.on(Events.EPOCH_COMPLETED)
    def run_validation(engine):
        if engine.state.epoch % val_interval == 0: 
            train_evaluator.run(config.train_eval_loader)
            evaluator.run(config.val_loader)

    # Log train eval metrics:
    tb_logger.attach(train_evaluator,
                     log_handler=tb_output_handler(tag="training", 
                                                   metric_names=list(val_metrics.keys()),
                                                   another_engine=trainer),
                     event_name=Events.EPOCH_COMPLETED)

    plx_logger.attach(train_evaluator,
                      log_handler=plx_output_handler(tag="training", 
                                                     metric_names=list(val_metrics.keys()),
                                                     another_engine=trainer),
                       event_name=Events.EPOCH_COMPLETED)

    # Log val metrics:
    tb_logger.attach(evaluator,
                     log_handler=tb_output_handler(tag="validation", 
                                                   metric_names=list(val_metrics.keys()),
                                                   another_engine=trainer),
                     event_name=Events.EPOCH_COMPLETED)

    plx_logger.attach(evaluator,
                      log_handler=plx_output_handler(tag="validation", 
                                                     metric_names=list(val_metrics.keys()),
                                                     another_engine=trainer),
                       event_name=Events.EPOCH_COMPLETED)

    # Log predictions:
    tb_logger.attach(evaluator, 
                     log_handler=predictions_gt_images_handler(n_images=15, another_engine=trainer),
                     event_name=Events.EPOCH_COMPLETED)

    # Log optimizer parameters
    tb_logger.attach(trainer,
                     log_handler=tb_optimizer_params_handler(optimizer, param_name="lr"),
                     event_name=Events.ITERATION_COMPLETED)

    # Store the best model
    def default_score_fn(engine):
        score = engine.state.metrics['mIoU']
        return score

    score_function = default_score_fn if not hasattr(config, "score_function") else config.score_function

    best_model_handler = ModelCheckpoint(dirname=path, 
                                         filename_prefix="best",
                                         n_saved=3,
                                         score_function=score_function)
    evaluator.add_event_handler(Events.COMPLETED, best_model_handler, {'model': model, })

    # Add early stopping
    if hasattr(config, "es_patience"):
        es_handler = EarlyStopping(patience=config.es_patience, score_function=score_function, trainer=trainer)    
        evaluator.add_event_handler(Events.COMPLETED, es_handler)

    trainer.run(config.train_loader, max_epochs=config.num_epochs)
