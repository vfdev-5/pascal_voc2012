import torch

try:
    from polyaxon_client.tracking import Experiment
except ImportError:
    raise RuntimeError("This contrib module requires polyaxon-client to be installed.")

from ignite.engine import Events


__all__ = ['PolyaxonLogger', 'output_handler']


# TODO: Move the mapping to State
MAP_EVENT_TO_STATE_ATTR = {
    Events.ITERATION_STARTED: "iteration",
    Events.ITERATION_COMPLETED: "iteration",
    Events.EPOCH_STARTED: "epoch",
    Events.EPOCH_COMPLETED: "epoch",
    Events.STARTED: "epoch",
    Events.COMPLETED: "epoch"
}


def output_handler(tag, metric_names=None, output_transform=None, another_engine=None):
    if metric_names is not None and not isinstance(metric_names, list):
        raise TypeError("metric_names should be a list, got {} instead.".format(type(metric_names)))

    if output_transform is not None and not callable(output_transform):
        raise TypeError("output_transform should be a function, got {} instead."
                        .format(type(output_transform)))

    if output_transform is None and metric_names is None:
        raise ValueError("Either metric_names or output_transform should be defined")

    def output_handler_wrapper(engine, writer, state_attr):

        metrics = {}
        if metric_names is not None:
            if not all(metric in engine.state.metrics for metric in metric_names):
                # -> Maybe display a warning ?
                pass
            else:
                metrics.update({name: engine.state.metrics[name] for name in metric_names})

        if output_transform is not None:
            output_dict = output_transform(engine.state.output)

            if not isinstance(output_dict, dict):
                output_dict = {"output": output_dict}

            metrics.update({name: value for name, value in output_dict.items()})

        state = engine.state if another_engine is None else another_engine.state
        global_step = getattr(state, state_attr)

        writer.log_metrics(step=global_step, **{"{}/{}".format(tag, k): v for k, v in metrics.items()})

    return output_handler_wrapper


class PolyaxonLogger(object):
    """
    Polyaxon tracking client handler to log parameters and metrics during the training and validation.

    """

    def __init__(self):
        self.experiment = Experiment()

    def __getattr__(self, attr):        
        def wrapper(*args, **kwargs):
            return getattr(self.experiment, attr)(*args, **kwargs)
        return wrapper

    def attach(self, engine, log_handler, event_name):
        """Attach the logger to the engine and execute `log_handler` function at `event_name` events.

        Args:
            engine (Engine): engine object.
            log_handler (callable): a logging handler to execute
            event_name: event to attach the logging handler to. Valid events are from :class:`~ignite.engine.Events`
                or any `event_name` added by :meth:`~ignite.engine.Engine.register_events`.

        """
        engine.add_event_handler(event_name, log_handler, self.experiment, MAP_EVENT_TO_STATE_ATTR[event_name])
