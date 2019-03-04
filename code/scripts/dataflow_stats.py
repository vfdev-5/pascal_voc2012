import random

import numpy as np

import torch

import ignite
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers import ProgressBar
from ignite.utils import to_onehot

from polyaxon_client.tracking import get_outputs_path

from custom_ignite.contrib.config_runner.utils import set_seed

from custom_ignite.contrib.handlers import TensorboardLogger
from custom_ignite.contrib.handlers.tensorboard_logger import output_handler as tb_output_handler

from custom_ignite.contrib.handlers import PolyaxonLogger 
# from custom_ignite.contrib.handlers.polyaxon_logger import output_handler as plx_output_handler


def remove_handler(engine, handler, event_name):
    assert event_name in engine._event_handlers    
    engine._event_handlers[event_name] = [(h, args, kwargs) 
                                          for h, args, kwargs in engine._event_handlers[event_name] 
                                            if h != handler]


def run(config, logger):
    
    plx_logger = PolyaxonLogger()

    set_seed(config.seed)

    plx_logger.log_params(**{
        "seed": config.seed,
        "batch_size": config.batch_size,

        "pytorch version": torch.__version__,
        "ignite version": ignite.__version__,
        "cuda version": torch.version.cuda
    })

    device = config.device
    non_blocking = config.non_blocking
    prepare_batch = config.prepare_batch

    def stats_collect_function(engine, batch):

        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)

        y_ohe = to_onehot(y.reshape(-1), config.num_classes)
        
        class_distrib = y_ohe.mean(dim=0).cpu()
        class_presence = (class_distrib > 1e-3).cpu().float()
        num_classes = (class_distrib > 1e-3).sum().item() 

        engine.state.class_presence += class_presence
        engine.state.class_presence -= (1 - class_presence)

        return {
            "class_distrib": class_distrib,
            "class_presence": engine.state.class_presence,
            "num_classes": num_classes
        }

    stats_collector = Engine(stats_collect_function)
    ProgressBar(persist=True).attach(stats_collector)

    @stats_collector.on(Events.STARTED)
    def init_vars(engine):
        engine.state.class_presence = torch.zeros(config.num_classes)

    log_dir = get_outputs_path()
    if log_dir is None:
        log_dir = "output"

    tb_logger = TensorboardLogger(log_dir=log_dir)

    tb_handler = tb_output_handler(tag="training", output_transform=lambda x: x)
    tb_logger.attach(stats_collector,
                     log_handler=tb_handler,
                     event_name=Events.ITERATION_COMPLETED)

    stats_collector.run(config.train_loader, max_epochs=1)

    remove_handler(stats_collector, tb_handler, Events.ITERATION_COMPLETED)
    tb_logger.attach(stats_collector,
                     log_handler=tb_output_handler(tag="validation", output_transform=lambda x: x),
                     event_name=Events.ITERATION_COMPLETED)

    stats_collector.run(config.val_loader, max_epochs=1)
