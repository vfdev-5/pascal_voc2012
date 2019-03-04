
import click

from custom_ignite.contrib.config_runner.runner import run_script


@click.command()
@click.argument('script_filepath', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.argument('config_filepath', type=click.Path(exists=True, file_okay=True, dir_okay=False))
def command(script_filepath, config_filepath):
    """Method to run experiment (defined by a script file)

    Args:
        script_filepath (str): input script filepath
        config_filepath (str): input configuration filepath        

    """
    run_script(script_filepath, config_filepath)


if __name__ == "__main__":
    command()
