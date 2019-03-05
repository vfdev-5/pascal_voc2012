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

from custom_ignite.metrics import ConfusionMatrix, IoU, mIoU, Accuracy, Precision, Recall

from custom_ignite.contrib.handlers import TensorboardLogger
from custom_ignite.contrib.handlers.tensorboard_logger import output_handler as tb_output_handler, \
    optimizer_params_handler as tb_optimizer_params_handler, weights_scalar_handler as tb_weights_scalar_handler, \
    grads_scalar_handler as tb_grads_scalar_handler

from custom_ignite.contrib.handlers import PolyaxonLogger 
from custom_ignite.contrib.handlers.polyaxon_logger import output_handler as plx_output_handler

from custom_ignite.contrib.handlers.time_profiler import BasicTimeProfiler

from utils import predictions_gt_images_handler


def run(config, logger):
    
    plx_logger = PolyaxonLogger()

    path = get_outputs_path()
    if path is None:
        # Setup local output
        import os
        from datetime import datetime
        assert 'OUTPUT_PATH' in os.environ, "When running locally, 'OUTPUT_PATH' envvar should be defined"
        path = os.path.join(os.environ['OUTPUT_PATH'], 
            "{}-{}".format(datetime.now().strftime("%Y%m%d-%H%M%S"), config.config_filepath.stem))

    set_seed(config.seed)

    device = config.device
    use_fp16 = config.use_fp16 if hasattr(config, "use_fp16") else False

    plx_logger.log_params(**{
        "seed": config.seed,
        "fp16": use_fp16,
        "batch_size": config.batch_size,
        "model": config.model.__class__.__name__,

        "pytorch version": torch.__version__,
        "ignite version": ignite.__version__,
        "cuda version": torch.version.cuda
    })

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

    if hasattr(config, "use_time_profiling") and config.use_time_profiling:
        train_time_profiler = BasicTimeProfiler()
        train_time_profiler.attach(trainer)

    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
    # Checkpoint training
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

    cm_metric = ConfusionMatrix(num_classes=num_classes, output_transform=output_transform)

    val_metrics = {
        "mAcc": Accuracy(cm_metric),
        "mPr": Precision(cm_metric),
        "mRe": Recall(cm_metric),
        "IoU": IoU(cm_metric),
        "mIoU": mIoU(cm_metric),
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
    
    if hasattr(config, "use_time_profiling") and config.use_time_profiling:
        train_eval_time_profiler = BasicTimeProfiler()
        train_eval_time_profiler.attach(train_evaluator)
        val_time_profiler = BasicTimeProfiler()
        val_time_profiler.attach(evaluator)

    ProgressBar(persist=True, desc="Train Evaluation").attach(train_evaluator)
    ProgressBar(persist=True, desc="Val Evaluation").attach(evaluator)

    log_dir = get_outputs_path()
    if log_dir is None:
        log_dir = path

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
        lr_scheduler = config.lr_scheduler
        if isinstance(lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
            trainer.add_event_handler(Events.ITERATION_STARTED, lambda engine: lr_scheduler.step())
        else:
            trainer.add_event_handler(Events.ITERATION_STARTED, config.lr_scheduler)

    # @trainer.on(Events.STARTED)
    # def warmup_cudnn(engine):
    #     logger.info("Warmup CuDNN on random inputs")
    #     for _ in range(5):
    #         for size in [batch_size, len(test_loader.dataset) % batch_size]:
    #             warmup_cudnn(model, criterion, size, config)

    val_interval = config.val_interval if not config.debug else 1    
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
                     log_handler=predictions_gt_images_handler(n_images=15, 
                                                               single_img_size=(256, 256), 
                                                               another_engine=trainer),
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

    def empty_cuda_cache(engine):
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    trainer.add_event_handler(Events.EPOCH_COMPLETED, empty_cuda_cache)
    evaluator.add_event_handler(Events.COMPLETED, empty_cuda_cache)
    train_evaluator.add_event_handler(Events.COMPLETED, empty_cuda_cache)

    trainer.run(config.train_loader, max_epochs=config.num_epochs)

    if hasattr(config, "use_time_profiling") and config.use_time_profiling:
        print("\n--- Training ---")
        msg = train_time_profiler.print_results(train_time_profiler.get_results())
        with open(os.path.join(path, "train_time_profiling.log"), 'w') as h:
            h.write(msg)

        print("\n--- Train evaluation ---")
        msg = train_eval_time_profiler.print_results(train_eval_time_profiler.get_results())
        with open(os.path.join(path, "train_eval_time_profiling.log"), 'w') as h:
            h.write(msg)

        print("\n--- Validation ---")
        msg = val_time_profiler.print_results(val_time_profiler.get_results())
        with open(os.path.join(path, "val_time_profiling.log"), 'w') as h:
            h.write(msg)
