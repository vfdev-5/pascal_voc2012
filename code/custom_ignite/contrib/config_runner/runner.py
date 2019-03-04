
import os
import sys
import logging

try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path


from custom_ignite.contrib.config_runner.utils import load_module, setup_logger, set_seed
        

def run_script(script_filepath, config_filepath):
    """Method to run experiment (defined by a script file)

    Args:
        script_filepath (str): input script filepath
        config_filepath (str): input configuration filepath

    """
    # Add config path and current working directory to sys.path to correctly load the configuration
    sys.path.insert(0, Path(script_filepath).resolve().parent.as_posix())
    sys.path.insert(0, Path(config_filepath).resolve().parent.as_posix())
    sys.path.insert(0, os.getcwd())

    module = load_module(script_filepath)

    if "run" not in module.__dict__:
        raise RuntimeError("Script file '{}' should contain a method `run(config, **kwargs)`".format(script_filepath))

    exp_name = module.__name__
    run_fn = module.__dict__['run']

    if not callable(run_fn):
        raise RuntimeError("Run method from script file '{}' should callable function".format(script_filepath))

    # Setup configuration
    config = load_module(config_filepath)

    config.config_filepath = Path(config_filepath)
    config.script_filepath = Path(script_filepath)

    logger = logging.getLogger(exp_name)
    log_level = logging.INFO
    setup_logger(logger, log_level)

    try:
        run_fn(config, logger)
    except KeyboardInterrupt:
        logger.info("Catched KeyboardInterrupt -> exit")
    except Exception as e:  # noqa
        logger.exception("")
        exit(1)


if __name__ == "__main__":
    # To run profiler
    assert len(sys.argv) == 3
    run_script(sys.argv[1], sys.argv[2])
