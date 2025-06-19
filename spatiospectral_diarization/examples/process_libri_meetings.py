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
from eval.spatio_spectral.utils import synchronize
from lazy_dataset.database import  JsonDatabase
from dlp_mpi.collection import NestedDict

from spatiospectral_diarization.spatio_spectral_pipeline import SpatioSpectralDiarization

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
def main(_config, _run, db_json, datasets, setup, semistatic):
    """
    Main function to run the spatiospectral diarization pipeline on LibriWASN or LibriCSS datasets.
    """


    # Load the database
    #TODO db =
    db = JsonDatabase(db_json)


    # Initalize the Diarization pipeline
    pipeline = SpatioSpectralDiarization(return_intervals=True)


    # Iterate over the datasets
    for dataset_name in _config['datasets']:
        dataset = db.get_dataset(dataset_name)

        if semistatic:
            prepare_dataset = dataset_preparation_semistatic(dataset_name, setup)
        else:
            prepare_dataset = dataset_preparation(dataset_name, setup)
        # Prepare current dataset
        dataset = dataset.map(prepare_dataset)
        diarization_estimates = NestedDict()

        # Process the individual meetings
        for meeting in dlp_mpi.split_managed(dataset, allow_single_worker=True):
            # Get the audio data
            session_id = meeting['session_id']
            audio_data = meeting['audio_data']

            # Run the diarization pipeline
            dia_output = pipeline(
                audio_data=audio_data,
                session_id=meeting['session_id'],
                dataset=dataset_name,
                debug=_config['debug']
            )

            # Convert diarization result back to sample resolution
            diarization_estimates[session_id] = dia_output['diarization_estimate']
        # Aggregate the diarization estimates (if MPI is used)
        diarization_estimates.gather()


        output_path = Path(_config['experiment_dir']) / f"{dataset_name}_estimates.rttm"
        pb.array.interval.to_rttm(diarization_estimates, output_path)



if __name__ == '__main__':
    ex.run_commandline()
