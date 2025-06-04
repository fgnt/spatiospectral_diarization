import copy
import dlp_mpi
import paderbox as pb
from einops import rearrange
import numpy as np
from dlp_mpi.collection import NestedDict
from paderbox.io.new_subdir import NameGenerator
from spatiospectral_diarization.embedding_based_clustering import embeddings_hdbscan_clustering
from spatiospectral_diarization.extraction.mask_estimation import get_dominant_time_frequency_mask, \
    extract_segment_stft_and_context, compute_smoothed_scms, compute_steering_and_similarity_masks, \
    resolve_mask_ambiguities, cacgmm_mask_refinement
from spatiospectral_diarization.spatial_diarization.diarize import spatial_diarization
from speaker_reassignment.tcl_pretrained import PretrainedModel
import padertorch as pt
from spatiospectral_diarization.extraction.beamformer import time_varying_mvdr
from sacred import Experiment, commands
from sacred.observers import FileStorageObserver
from pathlib import Path
import logging
from lazy_dataset.database import JsonDatabase
from spatiospectral_diarization.spatial_diarization.utils import get_position_candidates
from spatiospectral_diarization.spatial_diarization.cluster import temporally_constrained_clustering
from spatiospectral_diarization.spatial_diarization.utils import (channel_wise_activities,
                                                                  convert_to_frame_wise_activities)
from spatiospectral_diarization.utils import setup_logger, load_signals, select_channels, merge_and_extract_segments, \
    postprocess_and_get_activities, extract_embeddings, assign_estimated_activities, dump_rttm

ex = Experiment('libriwasn_pipeline_v4')
ex.observers.append(FileStorageObserver.create('runs'))

MAKEFILE = """
SHELL := /bin/bash

evaluate:
\tmpiexec --use-hwthread-cpus -np 8 python -m spatiospectral_diarization.spatiospectral_pipeline with config.json

debug:
\tpython -m spatiospectral_diarization.spatiospectral_pipeline with config.json --pdb

"""

BATCHFILE_TEMPLATE_EVAL = """#!/bin/bash
#SBATCH -t 4:00:00 
#SBATCH --mem-per-cpu 8G
#SBATCH -J libriwasn_spatiospectral_{nickname}
#SBATCH --cpus-per-task 1
#SBATCH -A hpc-prf-nt2
#SBATCH -p normal
#SBATCH -n 61
#SBATCH --output {nickname}_eval_%j.out
#SBATCH --error {nickname}_eval_%j.err

srun python -m {main_python_path} with config.json

"""

@ex.config
def config():
    json_path = '/net/vol/tgburrek/db/libriwasn_netdb.json'  #  "/net/vol/jenkins/jsons/notsofar.json" #
    dsets = ['libricss','libriwasn200', 'libriwasn800'] # [libricss, libriwasn200, libriwasn800] # ["train_set_240130.1_train"] #

    setup = 'compact'  # (compact, distributed)
    channels = 'set1' # [set1, set2, set3, all]
    debug = False
    experiment_dir = None
    if experiment_dir is None:
        experiment_dir = pt.io.get_new_storage_dir(
            'libriwasn_spatiospectral',
            id_naming=NameGenerator(('adjectives', 'colors', 'animals')),
        )


@ex.named_config
def distributed():
    setup = 'distributed'
    channels = 'set1'
    dsets = ['libriwasn200', 'libriwasn800']

@ex.named_config
def libri():
    json_path = '/net/vol/tgburrek/db/libriwasn_netdb.json'
    dsets = ['libricss', 'libriwasn200','libriwasn800']
    setup = 'compact'  # (compact, distributed)
    channels = 'set1'  # [set1, set2, set3, all]
    debug = False
    tmp_class_th_compact = 0.75
    noctua2 = False
    noctua1 = False
    experiment_dir = None
    if experiment_dir is None:
        experiment_dir = pt.io.get_new_storage_dir(
            'libriwasn_spatiospectral',
            id_naming=NameGenerator(('adjectives', 'colors', 'animals')),
        )

@ex.named_config
def noctua():
    noctua2=True
    json_path='/scratch/hpc-prf-nt1/cord/jsons/libriwasn_netdb.json'
    # json_path='/scratch/hpc-prf-nt2/deegen/deploy/forschung/DiariZen/notsofar.json'

@ex.named_config
def noctua1():
    noctua1=True
    json_path='/scratch-n2/hpc-prf-nt1/cord/jsons/libriwasn_netdb.json'

@ex.command(unobserved=True)
def init(_run, _config):
    """
    Dumps the configuration into the evaluation dir.
    """
    experiment_dir = Path(_config['experiment_dir'])
    makefile_path = Path(experiment_dir) / "Makefile"
    commands.print_config(_run)

    makefile_path.write_text(MAKEFILE)
    pt.io.dump_config(
        copy.deepcopy(_config),
        Path(experiment_dir) / 'config.json'
    )
    batchfile_path = Path(experiment_dir) / "eval.sh"
    batchfile_path.write_text(
        BATCHFILE_TEMPLATE_EVAL.format(
            main_python_path=pt.configurable.resolve_main_python_path(),
            nickname=experiment_dir.name,
        )
    )
    commands.print_config(_run)
    print(f'Evaluation directory: {experiment_dir}')
    print(f'Start evaluation with:')
    print()
    print(f'cd {experiment_dir}')
    print(f'make evaluate')

@ex.capture
def spatio_spectral_diarization(json_path, dsets, setup, channels, experiment_dir, noctua2, noctua1, debug,
                             kernel_size_scm_smoothing=3, eig_val_ratio_th=.9, min_cl_segment=3, frame_size_gcc=4096,
                             frame_shift_gcc=1024, avg_len_gcc=4, max_temp_dist_cl=16, min_srp_peak_rato=.5,
                             dilation_len_spatial=32001, dilation_len_spatial_add=8001, th=.3,
                             act_th=.2, act_th2=.3, erosion_len_beam=127, dilation_len_beam=127, min_len=32,
                             k_min=10, k_max=225, max_offset=0, context=48000, fft_size=1024):
    frame_shift = fft_size // 4
    if setup == 'distributed':
        distributed = True
        max_diff_gcc = 2.
        search_range_gcc = 200
        f_max_gcc = 3500
        max_diff_tmp_cl = 2.
    else:
        distributed = False
        max_diff_gcc = 1.
        search_range_gcc = 5
        f_max_gcc = None
        max_diff_tmp_cl = 0.75
    logger = setup_logger(log_level=logging.DEBUG)
    logger.info("Starting Spatio-Spectral Pipeline")
    db = JsonDatabase(json_path)
    embed_extractor = PretrainedModel()
    channels = select_channels(channels)

    for dset in dsets:
        spatial = NestedDict()
        est_diarization_mm = NestedDict()
        est_diarization_wo_mm = NestedDict()
        dataset = db.get_dataset(dset)
        if dlp_mpi.IS_MASTER:
            logger.info(f"Dataset length: {len(dataset)}")

        for session in dlp_mpi.split_managed(dataset, allow_single_worker=True):
            session_name = session['example_id']
            """ Load the audio signals and get VADs"""
            sigs = load_signals(session, channels, setup, dset, logger)
            voice_activity = channel_wise_activities(sigs)
            frame_wise_voice_activity = convert_to_frame_wise_activities(
                voice_activity, frame_size=frame_size_gcc, frame_shift=frame_shift_gcc
            )
            if debug:
                logger.info(f"{session_name}")
            sigs_stft = pb.transform.stft(
                sigs, frame_size_gcc, frame_shift_gcc, pad=False, fading=False
            )
            dominant = get_dominant_time_frequency_mask(sigs_stft)
            candidates = get_position_candidates(
                sigs_stft, frame_wise_voice_activity, dominant,
                max_diff=max_diff_gcc, search_range=search_range_gcc,
                f_max=f_max_gcc, avg_len=avg_len_gcc, distributed=distributed
            )
            temp_diary = temporally_constrained_clustering(
                    candidates, max_dist=max_diff_tmp_cl,
                    max_temp_dist=max_temp_dist_cl, peak_ratio_th=min_srp_peak_rato
            )
            temp_diary = temp_diary[::-1] #sort segments that segments that start earlier come first
            segments, seg_tdoas = merge_and_extract_segments(temp_diary, sigs, avg_len_gcc, min_cl_segment, distributed, max_diff_tmp_cl)
            ###################################################################
            """ Spatial Diarization """
            est_activities_spatial, labels, num_spk = spatial_diarization(distributed, seg_tdoas, segments, sigs,
                                                                          dilation_len_spatial, dilation_len_spatial_add)
            ###################################################################
            """Spatio-spectral Diarization"""
            embeddings = []
            embeddings_mm = []
            seg_boundaries = []
            seg_boundaries_mm = []
            sigs_stft_complete = pb.transform.stft(
                sigs, fft_size, frame_shift, pad=False, fading=False
            )
            for seg_idx in range(len(segments)):
                sigs_stft, tdoas_segment, activities, onset, offset = extract_segment_stft_and_context(seg_idx, segments,
                                                                    sigs, sigs_stft_complete, seg_tdoas, max_offset,
                                                                    frame_shift, fft_size, context, max_diff_tmp_cl, logger)
                scms, dominant = compute_smoothed_scms(sigs_stft, kernel_size_scm_smoothing, eig_val_ratio_th)

                k = np.arange(fft_size // 2 + 1)
                """ Compute masks, postprocess the masks and activities"""
                masks, inst_scm = compute_steering_and_similarity_masks(sigs_stft, sigs, tdoas_segment, k, fft_size, th)
                masks = resolve_mask_ambiguities(masks, tdoas_segment, sigs, k, fft_size, inst_scm, dominant)
                masks, seg_acitivities, tdoas_reduced, phantom = postprocess_and_get_activities(masks, tdoas_segment,
                                                                                                k_min, k_max, act_th, min_len,
                                                                                                dilation_len_beam, erosion_len_beam,
                                                                                                logger, additional_dilate=False,
                                                                                                cacgmm_param=False, reduce_tdoas=True)
                if phantom:
                    continue # skip segments of phantom speakers

                """Mask estimation with CACGMM"""
                masks = cacgmm_mask_refinement(masks, sigs_stft, seg_acitivities, dominant, fft_size, logger)

                masks, seg_acitivities, _, phantom = postprocess_and_get_activities(masks, tdoas_segment,
                                                                                    k_min, k_max, act_th, min_len,
                                                                                    dilation_len_beam, erosion_len_beam,
                                                                                    logger, additional_dilate=True,
                                                                                    cacgmm_param=True, reduce_tdoas=False)
                if phantom:
                    continue # skip segments of phantom speakers
                """Beamform the signals using the CACGMM masks"""
                sig_segs, seg_onsets = time_varying_mvdr(sigs_stft, rearrange(masks, 's t f -> s f t'),
                                                         seg_acitivities.astype(bool), wpe=False)
                """ Extract embeddings for the CACGMM masked signals"""
                embeddings_mm, seg_boundaries_mm = extract_embeddings(embeddings_mm, seg_boundaries_mm, sig_segs,
                                                                      seg_onsets, embed_extractor, onset, frame_shift)

                ##################################################
                """ Compute masks, postprocess the masks and activities"""
                masks, inst_scm = compute_steering_and_similarity_masks(sigs_stft, sigs, tdoas_reduced, k, fft_size, th)
                masks = resolve_mask_ambiguities(masks, tdoas_reduced, sigs, k, fft_size, inst_scm, dominant)
                masks, seg_acitivities, tdoas_reduced, phantom = postprocess_and_get_activities(masks, tdoas_reduced,
                                                                                                k_min, k_max, act_th2, min_len,
                                                                                                dilation_len_beam, erosion_len_beam,
                                                                                                logger, additional_dilate=True,
                                                                                                cacgmm_param=False, reduce_tdoas=False)
                if phantom:
                    continue # skip segments of phantom speakers

                """Beamform the signals using the masks"""
                sig_segs, seg_onsets = time_varying_mvdr(sigs_stft, rearrange(masks, 's t f -> s f t'), seg_acitivities.astype(bool), wpe=False)
                """ Extract embeddings for signals without CACGMM"""
                embeddings, seg_boundaries = extract_embeddings(embeddings, seg_boundaries, sig_segs, seg_onsets,
                                                                embed_extractor, onset, frame_shift)

            """ Perform embeddings clustering for embeddigns without CACGMM"""
            labels, activities_red, embeddings_red = embeddings_hdbscan_clustering(embeddings, seg_boundaries)
            est_activities = assign_estimated_activities(labels, activities_red, embeddings_red, sigs)

            """ Perform embeddings clustering for CACGMM embeddigns"""
            labels, activities_red, embeddings_red = embeddings_hdbscan_clustering(embeddings_mm, seg_boundaries_mm)
            est_activities_mm = assign_estimated_activities(labels, activities_red, embeddings_red, sigs)

            ###################################################################
            """Get Array interval for each speaker and put into diary"""
            spatial[session_name] = {spk: pb.array.interval.ArrayInterval(act.astype(bool))
                                     for spk, act in enumerate(est_activities_spatial)}
            est_diarization_mm[session_name] = {spk: pb.array.interval.ArrayInterval(act.astype(bool))
                                                for spk, act in enumerate(est_activities_mm)}
            est_diarization_wo_mm[session_name] = {spk: pb.array.interval.ArrayInterval(act.astype(bool))
                                                   for spk, act in enumerate(est_activities)}

            del sigs_stft_complete, sigs, voice_activity, sigs_stft, frame_wise_voice_activity, dominant, scms, inst_scm
        spatial = spatial.gather()
        est_diarization_mm = est_diarization_mm.gather()
        est_diarization_wo_mm = est_diarization_wo_mm.gather()
        if dlp_mpi.IS_MASTER:
            dump_rttm(spatial, path=Path(experiment_dir) / f'diarization_estimates_spatial_{dset}.rttm')
            dump_rttm(est_diarization_mm, path=Path(experiment_dir) / f'diarization_estimates_mm_{dset}.rttm')
            dump_rttm(est_diarization_wo_mm, path=Path(experiment_dir) / f'diarization_estimates_wo_mm_{dset}.rttm')
        dlp_mpi.barrier()
    return

@ex.main
def main(_config, _run):
    '''experiment_dir = Path(_config['experiment_dir'])
    pt.io.dump_config(
        copy.deepcopy(_config),
        Path(experiment_dir) / 'config.json'
    )
    pb.io.dump_json(_config, Path(_config['experiment_dir']) / 'config.json')'''
    spatio_spectral_diarization()


if __name__ == '__main__':
    ex.run_commandline()