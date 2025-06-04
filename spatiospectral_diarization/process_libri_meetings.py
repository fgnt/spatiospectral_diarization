from pathlib import Path
import numpy as np
import copy
from einops import rearrange
import logging

from sacred import Experiment, commands
from sacred.observers import FileStorageObserver

import dlp_mpi
import padertorch as pt
import paderbox as pb
from lazy_dataset.database import  JsonDatabase
from dlp_mpi.collection import NestedDict

from spatiospectral_diarization import SpatioSpectralDiarization

experiment_name = 'spatiospectral_diarization_libriwasn'
ex = Experiment(experiment_name)
ex.observers.append(FileStorageObserver('runs'))


MAKEFILE_TEMPLATE = """
SHELL := /bin/bash

evaluate:
\tmpiexec --use-hwthread-cpus -np 8 python -m {main_python_path}e with config.json

debug:
\tpython -m {main_python_path} with config.json --pdb

"""

@ex.config
def config():
    """
    Configuration for applying the spatiospectral diarization pipeline to either LibriWASN or LibriCSS.
    """
    datasets = ['libriwasn200', 'libriwasn800', 'libricss']
    setup = 'compact'
    db_json = ''
    assert len(db_json)>0, "Please provide a valid path to the prepared JSON file of LibriWASN."
    debug = False
    # Create a new experiment folder under 'STORAGE_ROOT'/experiment_name
    experiment_dir = None
    if experiment_dir is None:
        experiment_dir = pt.io.get_new_subdir(
            experiment_name, consider_mpi=True)

    ex.observers.append(FileStorageObserver(
        Path(Path(experiment_dir) / 'sacred')
    ))
    semistatic = False

@ex.named_config
def distributed():
    """
    Configuration for distributed microphone setup.
    """
    setup = 'distributed'
    datasets = ['libriwasn200', 'libriwasn800']


@ex.named_config
def semistatic():
    """
    Configuration for distributed microphone setup.
    """
    semistatic = True


@ex.command(unobserved=True)
def init(_run, _config):
    """
    Dumps the configuration into the evaluation dir.
    """
    experiment_dir = Path(_config['experiment_dir'])
    makefile_path = Path(experiment_dir) / "Makefile"
    commands.print_config(_run)

    makefile_path.write_text(MAKEFILE_TEMPLATE.format(main_python_path=pt.configurable.resolve_main_python_path()))
    pt.io.dump_config(
        copy.deepcopy(_config),
        Path(experiment_dir) / 'config.json'
    )
    commands.print_config(_run)
    print(f'Evaluation directory: {experiment_dir}')
    print(f'Start evaluation with:')
    print()
    print(f'cd {experiment_dir}')
    print(f'make evaluate')


@ex.main
def main(_config, _run):
    """
    Main function to run the spatiospectral diarization pipeline on LibriWASN or LibriCSS datasets.
    """


    # Load the database
    #TODO db =
    db = JsonDatabase(_config['db_json'])

    for dataset_name in _config['datasets']:
        dataset = db.get_dataset(dataset_name)

        if dataset is None:

        db = db.filter(lambda x: x['setup'] == _config['setup'])
        db = db.filter(lambda x: x['semistatic'] == _config.get('semistatic', False))

        # Initialize the diarization pipeline

    spatiospectral_pipeline = SpatioSpectralDiarization()




if __name__ == '__main__':
    ex.run_commandline()
