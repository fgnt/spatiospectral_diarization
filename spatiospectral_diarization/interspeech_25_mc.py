import copy
import dlp_mpi
from dlp_mpi.collection import NestedDict
from paderbox.io.new_subdir import NameGenerator
from speaker_reassignment.tcl_pretrained import PretrainedModel
import padertorch as pt
from spatiospectral_diarization.extraction.beamformer import time_varying_mvdr
from libriwasn.mask_estimation.initialization import correlation_matrix_distance
from sacred import Experiment, commands,SETTINGS
from spatiospectral_diarization.spatial_diarization.utils import (
    channel_wise_activities,
    convert_to_frame_wise_activities
)
from copy import deepcopy
from pathlib import Path
from lazy_dataset.database import JsonDatabase
from paderwasn.synchronization.sro_estimation import DynamicWACD
from paderwasn.synchronization.utils import VoiceActivityDetector
from .sro_compensation.sync import compensate_for_sros
from sklearn.cluster import HDBSCAN, AgglomerativeClustering
from spatiospectral_diarization.spatial_diarization.utils import erode, dilate
import scipy
from spatiospectral_diarization.spatial_diarization.srp_phat import get_position_candidates
from spatiospectral_diarization.spatial_diarization.cluster import temporally_constrained_clustering
from spatiospectral_diarization.spatial_diarization.utils import (
    convert_to_frame_wise_activities
)
from padercontrib.database.iterator import PrefixCorrector
import itertools

from scipy.signal import find_peaks

from spatiospectral_diarization.spatial_diarization.utils import get_ch_pairs

from einops import rearrange

import numpy as np
from scipy.signal import fftconvolve
import paderbox as pb
from pb_bss.math.solve import stable_solve
from pb_bss.distribution.cacgmm import CACGMMTrainer
from pb_bss.permutation_alignment import DHTVPermutationAlignment
from pb_bss.extraction.beamformer import get_power_spectral_density_matrix
from pb_bss.extraction.beamformer import get_mvdr_vector_souden
from pb_bss.extraction.beamformer import blind_analytic_normalization
from pb_bss.extraction.beamformer_wrapper import get_gev_rank_one_estimate
from .utils import Kernel1D
from nara_wpe.wpe import wpe_v8

from spatiospectral_diarization.extraction.utils import get_sdrs, get_interference_segments
import matplotlib.pyplot as plt

ex = Experiment('libriwasn_pipeline_v3')

MAKEFILE = """
SHELL := /bin/bash

evaluate:
\tmpiexec --use-hwthread-cpus -np 8 python -m tcl.eval.spatio_spectral.interspeech_25_final with config.json

debug:
\tpython -m tcl.eval.spatio_spectral.interspeech_25_final with config.json --pdb

"""

BATCHFILE_TEMPLATE_EVAL = """#!/bin/bash
#SBATCH -t 4:00:00 
#SBATCH --mem-per-cpu 16G
#SBATCH -J libriwasn_spatiospectral_{nickname}
#SBATCH --cpus-per-task 1
#SBATCH -A hpc-prf-nt2
#SBATCH -p normal
#SBATCH -n 31
#SBATCH --output {nickname}_eval_%j.out
#SBATCH --error {nickname}_eval_%j.err
#SBATCH --mail-type ALL
#SBATCH --mail-user cord@nt.upb.de

srun python -m {main_python_path} with config.json

"""

@ex.config
def config():
    json_path = '/net/vol/tgburrek/db/libriwasn_netdb.json'
    dsets = ['libricss','libriwasn200', 'libriwasn800'] # [libricss, libriwasn200, libriwasn800]

    setup = 'compact'  # (compact, distributed)
    channels = 'set1' # [set1, set2, set3, all]
    debug = False
    noctua2 = False
    noctua1 = False
    experiment_dir = None
    if experiment_dir is None:
        experiment_dir = pt.io.get_new_storage_dir(
            'libriwasn_spatiospectral_mc',
            id_naming=NameGenerator(('adjectives', 'colors', 'animals')),
        )
@ex.named_config
def distributed():
    setup = 'distributed'
    channels = ['set1']
    dsets = ['libriwasn200', 'libriwasn800']

@ex.named_config
def noctua():
    noctua2=True
    json_path='/scratch/hpc-prf-nt1/cord/jsons/libriwasn_netdb.json'

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


def estimate_sros(sigs):
    sro_estimator = DynamicWACD()
    sros = []
    energy = np.sum(
        pb.array.segment_axis(sigs[0][sigs[0] > 0], 1024, 256, end='cut') ** 2,
        axis=-1
    )
    th = np.min(energy[energy > 0])
    vad = VoiceActivityDetector(10 * th, len_smooth_win=0)
    ref_act = vad(sigs[0])
    ref_act = np.array(dilate(pb.array.interval.ArrayInterval(ref_act), 3201))
    ref_act = np.array(erode(pb.array.interval.ArrayInterval(ref_act), 3201))
    for ch_id in range(1, len(sigs)):
        energy = np.sum(
            pb.array.segment_axis(
                sigs[ch_id][sigs[ch_id] > 0], 1024, 256, end='cut'
            ) ** 2, axis=-1
        )
        th = np.min(energy[energy > 0])
        vad = VoiceActivityDetector(3 * th, len_smooth_win=0)
        act = vad(sigs[ch_id])
        act = np.array(dilate(pb.array.interval.ArrayInterval(act), 3201))
        act = np.array(erode(pb.array.interval.ArrayInterval(act), 3201))
        sro = sro_estimator(sigs[ch_id], sigs[0], act, ref_act)
        sros.append(sro)
    return sros


def channel_wise_activities(sigs, frame_size=1024, frame_shift=256):
    activities = np.zeros_like(sigs, bool)
    for ch_id, sig in enumerate(sigs):
        energy = np.sum(
            pb.array.segment_axis(
                sig[sig > 0], frame_size, frame_shift, end='cut'
            ) ** 2,
            axis=-1
        )
        th = np.min(energy[energy > 0])
        vad = VoiceActivityDetector(7 * th, len_smooth_win=0)
        act = vad(sig)
        act = np.array(dilate(pb.array.interval.ArrayInterval(act), 3201))
        act = np.array(erode(pb.array.interval.ArrayInterval(act), 3201))
        activities[ch_id] = act[:len(sig)]
    return activities


def solve_permutation(activities, ref_activities):
    if len(ref_activities) < len(activities):
        ref_activities = np.pad(
            ref_activities,
            ((0, len(activities) - len(ref_activities)), (0, 0)),
            'constant'
        )
    elif len(ref_activities) > len(activities):
        activities = np.pad(
            activities,
            ((0, len(ref_activities) - len(activities)), (0, 0)),
            'constant'
        )
    assert len(ref_activities) == len(activities), (
    len(ref_activities), len(activities))

    overlaps = np.zeros((len(activities), len(activities)))
    for i, act in enumerate(activities):
        for j, ref_act in enumerate(ref_activities):
            overlaps[i, j] = np.sum(act == ref_act)
    '''costs = []
    for permutation in permutations(np.arange(len(ref_activities))):
        cost = 0
        for j, i in enumerate(permutation):
            cost += overlaps[i, j]
            j += 1
        costs.append(cost)
    costs = np.asarray(costs)'''
    _, best_permutation = scipy.optimize.linear_sum_assignment(overlaps.T, maximize=True)
    '''all_permuations = \
        [permutation
         for permutation in permutations(np.arange(len(activities)))]
    best_permutation = all_permuations[np.argmax(costs)]'''
    return np.asarray(best_permutation)


def time_varying_mvdr(
        sigs_stft, masks, activities, wpe=True,
        frame_size=1024, min_len=32, eps=1e-18
):
    target_act = activities[0]
    activity_intervals = \
            pb.array.interval.ArrayInterval(target_act).normalized_intervals

    sig_segments = []
    segment_onsets = []
    for on, off in activity_intervals:
        if off - on < min_len:
            continue
        if wpe:
            sigs_stft_beam = wpe_v8(
                rearrange(sigs_stft[:, on:off], 'd t f -> f d t')
            )
            sigs_stft_beam = rearrange(sigs_stft_beam, 'f d t -> d t f')
        else:
            sigs_stft_beam = sigs_stft[:, on:off].copy()
        scm_target = get_power_spectral_density_matrix(
            rearrange(sigs_stft_beam, 'c t f -> f c t') * masks[0, :, None,  on:off],
            normalize=False
        )
        scm_target /= off - on
        interference_scm = get_power_spectral_density_matrix(
            rearrange(sigs_stft_beam, 'c t f -> f c t') * (1 - masks[0, :, None,  on:off]),
            normalize=False
        )
        interference_scm += \
            1e-9 * np.eye(len(sigs_stft))[None]
        scm_target_ = \
            get_gev_rank_one_estimate(scm_target, interference_scm)
        segment_info = get_interference_segments(activities[:, on:off], 0, off-on, 8)
        interference_segments = []
        for s, (on_, of_, concurrent) in enumerate(segment_info):
            interference_mask = 1 - masks[..., on:off][0, :, None,  on_:of_]
            interference_scm = get_power_spectral_density_matrix(
                rearrange(sigs_stft_beam[:, on_:of_], 'c t f -> f c t') * interference_mask,
                normalize=False
            )
            interference_scm /= of_ - on_
            interference_scm += \
                1e-8 * np.eye(len(sigs_stft_beam))[None]
            interference_segments.append((on_, of_, interference_scm))
        all_sdrs = []
        for (on_, of_, interference_scm) in interference_segments:
            phi = stable_solve(interference_scm, scm_target)
            lambda_ = np.trace(phi, axis1=-1, axis2=-2)[..., None, None]
            if eps is None:
                eps = np.finfo(lambda_.dtype).tiny
            w_mat = phi / np.maximum(lambda_.real, eps)
            sdrs_segment = get_sdrs(w_mat, scm_target, interference_scm)
            all_sdrs.append(sdrs_segment)
        all_sdrs = np.asarray(all_sdrs)
        all_sdrs = np.min(all_sdrs, 0)
        ref_ch = np.argmax(all_sdrs.real)
        bf_output = np.zeros(
            (off - on, sigs_stft_beam.shape[-1]), np.complex128
        )
        for (on_, of_, interference_scm) in interference_segments:
            bf_vec = get_mvdr_vector_souden(
                scm_target, interference_scm, ref_channel=ref_ch
            )
            bf_vec = \
                blind_analytic_normalization(bf_vec, interference_scm)
            for l in range(on_, of_):
                bf_output[l] = \
                    np.einsum('fc, cf-> f', np.conj(bf_vec), sigs_stft_beam[:, l])
        enh_sig = pb.transform.istft(
            bf_output, size=frame_size, shift=frame_size//4,
            window_length=frame_size, fading=False, pad=False
        )
        sig_segments.append(enh_sig)
        onset = \
            pb.transform.module_stft.stft_frame_index_to_sample_index(
                on, window_length=frame_size, shift=frame_size // 4,
                pad=False, fading=False, mode='first'
            )
        segment_onsets.append(onset)
    return sig_segments, segment_onsets


def cos_dist(embed_1, embed_2):
    return (1 - embed_1 @ embed_2 / np.sqrt(embed_1 @ embed_1) / np.sqrt(embed_2 @ embed_2) ) / 2




def get_gcpsd(fft_seg, fft_ref_seg):
    cpsd = np.conj(fft_ref_seg) * fft_seg
    phat = np.abs(fft_seg) * np.abs(fft_ref_seg)
    gcpsd = cpsd / np.maximum(phat, 1e-9)
    return gcpsd


def get_position_candidates(
        sigs_stft, frame_wise_activities, dominant, f_min=125, f_max=3500,
        search_range=200, avg_len=4, num_peaks=5, sample_rate=16000,
        max_diff=2, max_diff_same_pos=5, max_concurrent=3,distributed=False
):
    #verbose = True
    verbose = False

    max_val=.8
    kernel_size_scm_smoothing=3
    eig_val_ratio_th=.95
    kernel_size_scm_smoothing = 3
    ups_fact = 10
    assert num_peaks >= max_concurrent
    num_chs = len(sigs_stft)
    fft_size = frame_size = (sigs_stft.shape[-1] -1) * 2
    if f_min is None:
        k_min = None
    else:
        k_min = int(np.round(f_min / (sample_rate / 2) * (fft_size // 2 + 1)))
    if f_max is None:
        k_max = None
    else:
        k_max = int(np.round(f_max / (sample_rate / 2) * (fft_size // 2 + 1)))
    ch_pairs = get_ch_pairs(num_chs)
    lags = np.arange(-search_range, search_range + 1 / ups_fact, 1 / ups_fact)
    candidates = []
    gcpsd_buffer = \
        np.zeros((len(ch_pairs), avg_len, frame_size // 2 + 1), np.complex128)
    if verbose:
        gcc_mem = [[] for k, (i, j) in enumerate(ch_pairs)]
    for l in range(frame_wise_activities.shape[-1]):
        for k, (i, j) in enumerate(ch_pairs):
            gcpsd_buffer[k] = np.roll(gcpsd_buffer[k], -1, axis=0)
            gcpsd_buffer[k, -1] = 0
        if np.sum(frame_wise_activities[:, l]) == 0:
            if verbose:
                for k, (ref_ch, ch) in enumerate(ch_pairs):
                    gcc_mem[k].append(np.zeros_like(lags))
            continue
        gccs = []
        peak_tdoas = []
        peaks = []
        if verbose:
            print(l)
        for k, (ref_ch, ch) in enumerate(ch_pairs):
            fft_seg = sigs_stft[ch, l]
            fft_ref_seg = sigs_stft[ref_ch, l]
            gcpsd = get_gcpsd(fft_seg, fft_ref_seg)
            gcpsd_buffer[k, -1] = gcpsd * dominant[l]
            avg_gcpsd = np.mean(gcpsd_buffer[k], 0)
            avg_gcpsd[avg_gcpsd > 0.5 / avg_len] /= np.abs(avg_gcpsd[avg_gcpsd > 0.5 / avg_len])
            if k_min is not None:
                avg_gcpsd[:k_min] = 0.
            if k_max is not None:
                avg_gcpsd[k_max:] = 0.
            avg_gcpsd = np.concatenate(
                [avg_gcpsd[:-1],
                 np.zeros((ups_fact - 1) * (len(avg_gcpsd) - 1) * 2),
                 np.conj(avg_gcpsd)[::-1][:-1]]
            )
            gcc = np.fft.ifftshift(np.fft.ifft(avg_gcpsd).real)
            search_area = \
                gcc[len(gcc)//2-search_range*ups_fact:len(gcc)//2+search_range*ups_fact+1]
            th = np.maximum(0.75 * np.max(search_area), 0)#2 * np.sqrt(np.mean(search_area[search_area > 0] ** 2))
            '''peaks_pair = [np.argmax(search_area),]
            phase = np.exp(-1j * 2* np.pi * np.arange(2049) / 4096 * lags[np.argmax(search_area)])
            m = np.ones(2049)
            m -= np.abs(np.angle(avg_gcpsd[:2049] * phase.conj())) < np.pi / 4
            m = np.maximum(m, 0)
            for cnt in range(num_peaks):
                avg_gcpsd_ = m * avg_gcpsd[:2049]
                avg_gcpsd_ = np.concatenate(
                    [avg_gcpsd_[:-1],
                     np.zeros((ups_fact - 1) * (len(avg_gcpsd_) - 1) * 2),
                     np.conj(avg_gcpsd_)[::-1][:-1]]
                )
                gcc_ = np.fft.ifftshift(np.fft.ifft(avg_gcpsd_).real)
                search_area_ = \
                    gcc_[len(gcc_)//2-search_range*ups_fact:len(gcc_)//2+search_range*ups_fact+1]
                peaks_pair.append(np.argmax(search_area_))
                phase = np.exp(-1j * 2* np.pi * np.arange(2049) / 4096 * lags[np.argmax(search_area_)])
                m -= np.abs(np.angle(avg_gcpsd[:2049] * phase.conj())) < np.pi / 4
                m = np.maximum(m, 0)'''

            if False:#verbose and k == 4:
                plt.plot(lags, search_area)
                plt.title(f'Frame {l}')
                plt.grid()
                plt.show()
            if verbose:
                s = search_area.copy()
                s[s < th ] = 0
                gcc_mem[k].append(s)
            peaks_pair, _ = find_peaks(search_area)
            peaks_pair = np.asarray(peaks_pair)
            peaks_pair = peaks_pair[search_area[peaks_pair] >= th]
            choice = np.argsort(search_area[peaks_pair])[::-1][:num_peaks]
            peaks_pair = peaks_pair[choice]
            peaks.append(peaks_pair)
            for p, peak in enumerate(peaks_pair):
                if p+1 > len(gccs):
                    peak_tdoas.append(-1000*np.ones((len(sigs_stft), len(sigs_stft))))
                    gccs.append(np.zeros((len(sigs_stft), len(sigs_stft))))
                peak_tdoas[p][ref_ch, ch] = lags[peak]
                peak_tdoas[p][ch, ref_ch] = -lags[peak]
                gccs[p][ref_ch, ch] = gccs[p][ch, ref_ch] = search_area[peak]
            if verbose:
                print(k, np.round(lags[np.asarray(peaks_pair)], 2))
        '''srps = []
        for combination in itertools.product(*[np.arange(len(p)) for p in peaks]):
            tdoas = np.zeros((num_chs, num_chs))
            for p, (r, s) in zip(combination, ch_pairs):
                tdoas[r, s] = peak_tdoas[p][r, s]
                tdoas[s, r] = peak_tdoas[p][s, r]
            srp = 0
            for p, (r, s) in zip(combination, ch_pairs):
                srp += gccs[p][r, s]
            valid = True
            for (i, j) in ch_pairs:
                for k in range(num_chs):
                    if i == k or j == k:
                        continue
                    if abs(tdoas[i, j] - tdoas[i, k] + tdoas[j, k]) >= max_diff:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                tdoas = np.asarray([peak_tdoas[p][r, s]
                                    for p, (r, s) in zip(combination, ch_pairs)])
                srps.append((tdoas, srp))'''
        srps = []
        for combination in itertools.product(*[np.arange(len(p)) for p in peaks]):
            t = np.zeros((len(sigs_stft), len(sigs_stft)))
            for k, (ref_ch, ch) in enumerate(ch_pairs):
                t[ref_ch, ch] = peak_tdoas[combination[k]][ref_ch, ch]
                t[ch, ref_ch] = - t[ref_ch, ch]
            valid = True
            for k, (ref_ch, ch) in enumerate(ch_pairs):
                for m in range(len(sigs_stft)):
                    if m== ref_ch or m== ch:
                        continue
                    if np.max(abs(t[m, ch] + t[ref_ch, m] - t[ref_ch, ch])) > max_diff:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                srp = 0
                taus = []
                for k, (ref_ch, ch) in enumerate(ch_pairs):
                    srp += gccs[combination[k]][ref_ch, ch]
                for ref_ch in range(len(sigs_stft)):
                    for ch in range(ref_ch+1, len(sigs_stft)):
                        taus.append(t[ref_ch, ch])
                #print('valid', np.round(taus, 2), srp)
                if distributed:
                    srps.append((taus,srp))
                elif np.any(np.abs(taus) >= .5):
                    srps.append((taus,srp))
            '''taus = []
            tdoas = np.zeros((len(sigs_stft)))
            for p, i in zip(combination, range(1, len(sigs_stft))):
                tdoas[i] = peak_tdoas[p][0, i].copy()
                taus.append(tdoas[i])
            srp = 0
            for p, j in zip(combination, range(1, len(sigs_stft))):
                srp += gccs[p][0, j]
            skip = False
            for (i, j) in ch_pairs:
                if 0 == i or 0 == j:
                    continue
                for pid, p in enumerate(peak_tdoas):
                    if abs(p[i, j] - tdoas[j] + tdoas[i]) <= max_diff:
                        taus.append(p[i, j])
                        srp += gccs[pid][i, j]
                        break
                else:
                    skip = True
                if skip:
                    break
            if skip:
                continue
            srps.append((taus, srp))'''
        srps = sorted(srps, key=lambda ex: ex[-1], reverse=True)
        spk_pos = []
        for i in range(max_concurrent):
            if len(srps) == 0:
                break
            new_pos = srps[0]
            spk_pos.append(new_pos)
            taus = new_pos[0].copy()
            to_keep = []
            for srp in srps[1:]:
                t, _ = srp
                if np.sum(abs(np.asarray(t) - np.asarray(taus)) <= .3) <= 2:
                    to_keep.append(srp)
            srps = to_keep
        #spk_pos = [(pos[0][:num_chs-1], pos[1])for pos in spk_pos]
        candidates.append((l, spk_pos))
        if verbose:
            print()
    if verbose:
        for gcc in gcc_mem:
            plt.imshow(np.asarray(gcc).T , interpolation='nearest', aspect='auto', origin='lower')
            plt.yticks(np.arange(0, len(gcc[0]), 10), np.round(lags[::10], 1))
            plt.show()
    return candidates


def temporally_constrained_clustering(
        candidates, max_dist=5, max_temp_dist=16, peak_ratio_th=.75
):
    max_peaks = []
    for c in candidates:
        if len(c):
            if len(c[-1]) == 1:
                max_peaks.append(c[-1][0][-1])
    srp_th = 0#np.mean(max_peaks) - 2 * np.std(max_peaks)
    diary = []
    for (frame_id, spk_pos) in candidates:
        if not len(spk_pos):
            continue
        (tdoas, srp) = spk_pos[0]
        if srp < srp_th:
            continue
        tdoas = np.asarray(tdoas)
        if not len(diary):
            diary.append(([tdoas], [frame_id], [srp]))
            continue
        else:
            for c, entry in enumerate(diary):
                ref_tdoas = entry[0][-1]
                if frame_id - entry[1][-1] > max_temp_dist:
                    #diary.append(([tdoas], [frame_id], [srp]))
                    #break
                    continue
                #if cos_dist(np.array(ref_tdoas), np.array(tdoas)) < .01:
                #if np.max(abs(ref_tdoas - tdoas)) <= max_dist:
                if np.linalg.norm(ref_tdoas - tdoas) <= max_dist and np.max(np.abs(ref_tdoas - tdoas)) <= .5:
                    entry[0].append(tdoas)
                    entry[1].append(frame_id)
                    entry[2].append(srp)
                    break
            else:
                diary.append(([tdoas], [frame_id], [srp]))
        ref_srp = srp
        if ref_srp < srp_th:
            continue
        for (tdoas, srp) in spk_pos[1:]:
            if srp / ref_srp < peak_ratio_th:
                continue
            if srp < srp_th:
                continue
            tdoas = np.asarray(tdoas)
            if not len(diary):
                diary.append(([tdoas], [frame_id], [srp], [tdoas]))
                continue
            else:
                for c, entry in enumerate(diary):
                    if frame_id - entry[1][-1] > max_temp_dist:
                        #diary.append(([tdoas], [frame_id], [srp]))
                        #break
                        continue
                    match = False
                    for x in range(len(entry[0])):
                        if frame_id - entry[1][x] > max_temp_dist:
                            #diary.append(([tdoas], [frame_id], [srp]))
                            #break
                            continue
                        #if np.sum(np.abs(tdoas - ref_tdoas) > max_dist) <= 1:
                        #if cos_dist(np.array(entry[0][x]), np.array(tdoas)) < .01:
                        #if np.max(abs(entry[0][x] - tdoas)) <= max_dist:
                        if np.linalg.norm(entry[0][x] - tdoas) <= max_dist and np.max(np.abs(entry[0][x] - tdoas)) <= .5:
                            entry[0].append(tdoas)
                            entry[1].append(frame_id)
                            entry[2].append(srp)
                            match = True
                            break
                    if match:
                        break
                else:
                    diary.append(([tdoas], [frame_id], [srp]))
    diary = sorted(diary, key=lambda x: x[1][0], reverse=True)
    #diary = [entry for entry in diary if len(entry[1])>=4]
    return diary

combinations = [(0,4),(1,5), (2,8), (3,9), (6,7)]


@ex.capture
def spatio_spectral_pipeline(json_path, dsets,setup, channels,experiment_dir, noctua1, noctua2,debug):
    if setup == 'distributed':
        distributed = True
    else:
        distributed = False

    #TODO: params
    frame_size_gcc = 4096
    frame_shift_gcc = 1024
    if distributed:
        max_diff_gcc = 2.
        search_range_gcc = 200
        f_max_gcc = 3500
        avg_len_gcc = 4
        max_diff_tmp_cl = 2.
        max_temp_dist_cl = 16
        min_srp_peak_rato = .5
    else:
        max_diff_gcc = 1.
        search_range_gcc = 5
        f_max_gcc = None
        avg_len_gcc = 4
        max_diff_tmp_cl = .75
        max_temp_dist_cl = 16
        min_srp_peak_rato = .5

    max_val_init = .8
    kernel_size_scm_smoothing = 3
    eig_val_ratio_th = .9
    min_cl_segment = 3
    erosion_len_spatial = 32001
    dilation_len_spatial = 32001
    dilation_len_spatial_add = 8001

    th = .3
    act_th = .2
    act_th2 = .3
    act_th_mm = .4
    erosion_len_beam = 127
    dilation_len_beam = 127
    min_len = 32
    verbose = False

    k_min = 10
    k_max = 225
    max_offset = 0
    context = 48000

    db = JsonDatabase(json_path)
    embed_extractor = PretrainedModel(
    )
    for dset in dsets:
        subset = dset
        diarization_estimates = NestedDict()
        diarization_estimates_spatial = NestedDict()
        spatial = NestedDict()
        est_diarization_mm = NestedDict()
        est_diarization_wo_mm = NestedDict()
        embeddings_per_session = NestedDict()
        onsets_per_session = NestedDict()
        offsets_per_session = NestedDict()
        diarization_targets = NestedDict()
        dataset = db.get_dataset(dset)
        if noctua2:
            pc = PrefixCorrector(old_prefix='/net/db/', new_prefix='/scratch/hpc-prf-nt1/cord/data/')
            dataset = dataset.map(pc)
        if noctua1:
            pc = PrefixCorrector(old_prefix='/net/db/', new_prefix='/scratch-n2/hpc-prf-nt1/cord/data/')
            dataset = dataset.map(pc)

        channels = np.array([1, 3, 4, 6])

        if debug:
            dataset = list(dataset[-4:]) + list(dataset[10:12])
        all_combinations = []
        for offset in range(0,60,10):
            for c1, c2 in combinations:
                all_combinations.append((c1+offset, c2+offset))
        for combination in dlp_mpi.split_managed(all_combinations):
            session_1 = dataset[combination[0]]
            session_2 = dataset[combination[1]]
            session_name = session_1['example_id'] + '_' + session_2['example_id']
            if setup == 'compact':
                if session_1['dataset'] == 'libricss':
                    sigs_1 = pb.io.load_audio(session_1['audio_path']['observation'])[channels]
                    sess_1_len = sigs_1.shape[-1]
                    sigs_2 = pb.io.load_audio(session_2['audio_path']['observation'])[channels]
                    sigs = np.concatenate((sigs_1, sigs_2), axis=-1)
                    del sigs_1, sigs_2
                else:
                    sigs_1 = pb.io.load_audio(session_1['audio_path']['observation']['asnupb7'])
                    sess_1_len = sigs_1.shape[-1]
                    sigs_2 = pb.io.load_audio(session_2['audio_path']['observation']['asnupb7'])
                    sigs = np.concatenate((sigs_1, sigs_2), axis=-1)
                    del sigs_1, sigs_2
            elif setup == 'distributed':
                #session_1:
                sig0 = pb.io.load_audio(session_1['audio_path']['observation']['Pixel6a'])
                sig1 = pb.io.load_audio(session_1['audio_path']['observation']['Pixel6b'])

                sig2 = pb.io.load_audio(session_1['audio_path']['observation']['Pixel7'])
                sig3 = -pb.io.load_audio(session_1['audio_path']['observation']['Xiaomi'])

                min_len_sess = np.min([len(sig0), len(sig1), len(sig2), len(sig3)])
                sigs = np.vstack((sig0[:min_len_sess], sig1[:min_len_sess], sig2[:min_len_sess], sig3[:min_len_sess]))
                sros = estimate_sros(sigs)
                sigs_1 = compensate_for_sros(sigs, sros)
                sess_1_len = sigs_1.shape[-1]

                # session_2:
                sig0 = pb.io.load_audio(session_2['audio_path']['observation']['Pixel6a'])
                sig1 = pb.io.load_audio(session_2['audio_path']['observation']['Pixel6b'])

                sig2 = pb.io.load_audio(session_2['audio_path']['observation']['Pixel7'])
                sig3 = -pb.io.load_audio(session_2['audio_path']['observation']['Xiaomi'])

                min_len_sess = np.min([len(sig0), len(sig1), len(sig2), len(sig3)])
                sigs = np.vstack((sig0[:min_len_sess], sig1[:min_len_sess], sig2[:min_len_sess], sig3[:min_len_sess]))
                sros = estimate_sros(sigs)
                sigs_2 = compensate_for_sros(sigs, sros)

                sigs = np.concatenate((sigs_1, sigs_2), axis=-1)

            else:
                raise KeyError(f'Undefined Setup {setup}')


            ###################################################################
            voice_activity = channel_wise_activities(sigs)
            frame_wise_voice_activity = convert_to_frame_wise_activities(
                voice_activity, frame_size=frame_size_gcc, frame_shift=frame_shift_gcc
            )

            sigs_stft = pb.transform.stft(
                sigs, frame_size_gcc, frame_shift_gcc, pad=False, fading=False
            )
            dominant = np.zeros_like(sigs_stft[0], bool)
            eig_val_mem = np.zeros_like(sigs_stft[0])
            sigs_stft_ = np.pad(sigs_stft, ((0, 0), (1, 1), (0, 0)), mode='edge')
            kernel_size_scm_smoothing = 3
            eig_val_ratio_th = .9
            for i in range(1, sigs_stft_.shape[1]-1):
                scms = np.einsum('ctf, dtf -> fcd', sigs_stft_[:, i-1:i+2], sigs_stft_[:, i-1:i+2].conj())
                scms  = fftconvolve(
                    np.pad(
                        scms,
                        (
                            (kernel_size_scm_smoothing//2, kernel_size_scm_smoothing//2),
                            (0, 0),
                            (0, 0)),
                        mode='edge'
                    ),
                    1 / kernel_size_scm_smoothing * np.ones(
                        (kernel_size_scm_smoothing, len(sigs_stft), len(sigs_stft))
                    ),
                    axes=0,
                    mode='valid'
                )
                eig_vals, _ = np.linalg.eigh(scms)
                dominance = 1 - eig_vals[..., -2] / eig_vals[..., -1]
                dominant[i-1]  = (dominance >= eig_val_ratio_th)
                eig_val_mem[i-1] = eig_vals[..., -1]
            eig_val_th = 10 * np.min(eig_val_mem)
            dominant *= (eig_val_mem > eig_val_th)

            candidates = get_position_candidates(
                sigs_stft, frame_wise_voice_activity, dominant,
                max_diff=max_diff_gcc, search_range=search_range_gcc,
                f_max=f_max_gcc, avg_len=avg_len_gcc, distributed=distributed
            )

            fft_size = 1024
            frame_shift = fft_size // 4

            sigs_stft_complete = pb.transform.stft(
                sigs, fft_size, frame_shift, pad=False, fading=False
            )

            dominant = np.zeros_like(sigs_stft_complete[0], bool)
            eig_val_mem = np.zeros_like(sigs_stft_complete[0])
            sigs_stft_ = np.pad(sigs_stft_complete, ((0, 0), (1, 1), (0, 0)), mode='edge')
            for i in range(1, sigs_stft_.shape[1]-1):
                scms = np.einsum('ctf, dtf -> fcd', sigs_stft_[:, i-1:i+2], sigs_stft_[:, i-1:i+2].conj())
                scms  = fftconvolve(
                    np.pad(
                        scms,
                        (
                            (kernel_size_scm_smoothing//2, kernel_size_scm_smoothing//2),
                            (0, 0),
                            (0, 0)),
                        mode='edge'
                    ),
                    1 / kernel_size_scm_smoothing * np.ones(
                        (kernel_size_scm_smoothing, len(sigs_stft_complete), len(sigs_stft_complete))
                    ),
                    axes=0,
                    mode='valid'
                )
                eig_vals, _ = np.linalg.eigh(scms)
                dominance = 1 - eig_vals[..., -2] / eig_vals[..., -1]
                dominant[i-1]  = (dominance >= eig_val_ratio_th)
                eig_val_mem[i-1] = eig_vals[..., -1]
            eig_val_th = 10 * np.min(eig_val_mem)
            dominant *= (eig_val_mem > eig_val_th)
            dominant_complete = dominant.copy()

            temp_diary = temporally_constrained_clustering(
                    candidates, max_dist=max_diff_tmp_cl,
                    max_temp_dist=max_temp_dist_cl, peak_ratio_th=min_srp_peak_rato
            )
            temp_diary = temp_diary[::-1]

            temp_diary_ = deepcopy(temp_diary)
            seg_tdoas = []
            segments = []
            for i, entry in enumerate(temp_diary_):
                if not distributed:
                    if np.all(abs(np.median(entry[0], 0)) < .2):
                        continue
                if len(entry[1]) <= min_cl_segment:
                    continue

                med_tdoa = np.median(entry[0], 0)
                act = pb.array.interval.zeros(sigs.shape[-1])
                onset = np.maximum((np.min(entry[1]) - avg_len_gcc) * 1024, 0)
                offset = np.max(entry[1]) * 1024 + 4096
                act.add_intervals([slice(onset, offset), ])
                to_remove = []
                for o, other in enumerate(temp_diary_[i+1:]):
                    if np.linalg.norm(np.median(other[0], 0)- med_tdoa) <= max_diff_tmp_cl:
                        other_act = pb.array.interval.zeros(sigs.shape[-1])
                        onset = np.maximum((np.min(other[1]) - avg_len_gcc ) * 1024, 0)
                        offset = np.max(other[1]) * 1024 + 4096
                        other_act.add_intervals([slice(onset, offset), ])
                        if np.sum(np.array(act) * np.array(other_act)) > 0:
                            for t in other[0]:
                                entry[0].append(t)
                            for t in other[1]:
                                entry[1].append(t)
                            to_remove.append(i+1+o)
                for remove_id in to_remove[::-1]:
                    temp_diary_.pop(remove_id)
                med_tdoa = np.median(entry[0], 0)
                act = pb.array.interval.zeros(sigs.shape[-1])
                onset = np.maximum((np.min(entry[1]) - avg_len_gcc) * 1024, 0)
                offset = np.max(entry[1]) * 1024 + 4096
                act.add_intervals([slice(onset, offset), ])
                segments.append(act)
                seg_tdoas.append(med_tdoa)
            ###################################################################
            #TODO: Spatial Diarization
            if distributed:
                labels = AgglomerativeClustering(n_clusters=None, distance_threshold=5,linkage='single').fit_predict(seg_tdoas)
                min_samples = 3
                for i in range(np.max(labels)+ 1):
                    if np.sum(labels== i) < min_samples:
                        labels[labels == i] = -1
            else:
                labels = AgglomerativeClustering(n_clusters=None, distance_threshold=.25,linkage='single').fit_predict(seg_tdoas)
                min_samples = 3
                for i in range(np.max(labels)+ 1):
                    if np.sum(labels== i) < min_samples:
                        labels[labels == i] = -1

            labels = np.asarray(labels)
            mapping = {label: i for i, label in enumerate(set(labels[labels != -1]))}
            mapping[-1] = -1
            labels = [mapping[label] for label in labels]
            labels = np.asarray(labels)
            num_spk = np.max(labels) + 1
            est_activities = np.zeros((num_spk, sigs.shape[-1]), bool)

            for label, act in zip(labels, segments):
                if label == -1:
                    continue
                onset, offset = act.intervals[0]
                est_activities[label, onset:offset] = 1

            est_activities = [
                np.array(dilate(pb.array.interval.ArrayInterval(act),dilation_len_spatial))#Kernel1D(dilation_len_spatial, kernel=np.max)(act)
                for act in est_activities
            ]
            est_activities = [
                np.array(erode(pb.array.interval.ArrayInterval(act),dilation_len_spatial))#Kernel1D(erosion_len_spatial, kernel=np.min)(act)
                for act in est_activities
            ]
            est_activities = [
                np.array(dilate(pb.array.interval.ArrayInterval(act),dilation_len_spatial_add))#Kernel1D(dilation_len_spatial, kernel=np.max)(act)
                for act in est_activities
            ]

            est_activities = np.asarray(est_activities)

            est_activities_spatial = np.asarray(est_activities)



            ###################################################################
            #TODO: Spatio-spectal Diarization
            verbose = False
            embeddings = []
            embeddings_mm = []
            seg_boundaries = []
            seg_boundaries_mm = []
            for seg_idx in range(len(segments)):
                onset, offset = segments[seg_idx].intervals[0]
                act = pb.array.interval.zeros(sigs.shape[-1])
                act.add_intervals([slice(onset, offset), ])
                '''if np.sum(np.asarray(ref_seg) * np.asarray(act)) == 0:
                    continue'''
                if offset > max_offset:
                    max_offset = offset
                fft_size = 1024
                #print(onset, offset)
                onset = np.maximum(0, onset - int(context))
                offset = offset + int(context)
                if verbose:
                    pb.io.play(sigs[0, onset:offset])
                act = pb.array.interval.zeros(sigs.shape[-1])
                act.add_intervals([slice(onset, offset), ])
                tdoas_segment = [seg_tdoas[seg_idx]]
                #print(tdoas_segment[-1], segments[seg_idx].intervals[0])
                '''for b in range(len(test[seg_idx])):
                    print(temp_diary[seg_idx][0][b])
                print()'''
                activities = [act, ]

                for other_seg in range(len(segments)):
                    if seg_idx == other_seg:
                        continue
                    skip = False
                    for t in tdoas_segment:
                        if np.linalg.norm(t - seg_tdoas[other_seg]) <= max_diff_tmp_cl:#np.all(abs(t - seg_tdoas[other_seg]) < .5 ):
                            skip = True
                    if skip:
                        continue
                    other_act = np.asarray(segments[other_seg])
                    if np.sum(act * other_act) > 0:

                        tdoas_segment.append(seg_tdoas[other_seg])
                        activities.append(other_act)
                        #print(tdoas_segment[-1], segments[other_seg].intervals[0])
                        '''for b in range(len(test[other_seg])):
                            print(temp_diary[other_seg][0][b],test[other_seg][b])
                        print()'''
                if verbose:
                    print(np.round(tdoas_segment, 2))
                activities = [convert_to_frame_wise_activities(act[onset:offset]) for act in activities]
                onset = int(np.floor(onset // frame_shift))
                offset = int(np.ceil((offset - fft_size) / frame_shift))
                # print(onset, offset)
                sigs_stft = sigs_stft_complete[:, onset:offset].copy()#pb.transform.stft(sigs[:, onset:offset], fft_size, fft_size//4, pad=False, fading=False)
                sigs_stft = wpe_v8(
                    rearrange(sigs_stft, 'd t f -> f d t')
                )
                sigs_stft = rearrange(sigs_stft, 'f d t -> d t f')

                scms = np.einsum('c t f, d t f -> f t c d', sigs_stft, sigs_stft.conj())
                scms = [
                    fftconvolve(
                        np.pad(
                            freq,
                            (
                                (kernel_size_scm_smoothing//2, kernel_size_scm_smoothing//2),
                                (0, 0),
                                (0, 0)),
                            mode='reflect'
                        ),
                        1 / kernel_size_scm_smoothing * np.ones(
                            (kernel_size_scm_smoothing, len(sigs_stft), len(sigs_stft))
                        ),
                        axes=0,
                        mode='valid'
                    )
                    for freq in scms
                ]
                scms = rearrange(scms, 'f t c d -> t f c d')
                scms = [
                    fftconvolve(
                        np.pad(
                            freq,
                            (
                                (kernel_size_scm_smoothing//2, kernel_size_scm_smoothing//2),
                                (0, 0),
                                (0, 0)),
                            mode='reflect'
                        ),
                        1 / kernel_size_scm_smoothing * np.ones(
                            (kernel_size_scm_smoothing, len(sigs_stft), len(sigs_stft))
                        ),
                        axes=0,
                        mode='valid'
                    )
                    for freq in scms
                ]
                scms = rearrange(scms, 't f c d -> f t c d')
                scms = np.asarray(scms)
                eig_vals, eig_vects = np.linalg.eigh(scms)
                scms =  np.einsum('t f c, t f d -> f t c d', eig_vects[..., -1], eig_vects[..., -1].conj())
                eig_val_th = np.min(eig_vals) * 10
                dominance = 1 - eig_vals[..., -2] / eig_vals[..., -1]
                dominant = (dominance >= eig_val_ratio_th)
                dominant *= (eig_vals[..., -1] > eig_val_th)
                dominant = dominant.T
                if verbose:
                    plt.imshow((dominant).T , interpolation='nearest', aspect='auto', origin='lower')
                    plt.show()

                all_masks = []
                k = np.arange(fft_size // 2 + 1)

                all_masks = []
                inst_scm = np.einsum('ctf, dtf -> tfcd', sigs_stft, sigs_stft.conj())
                inst_scm /= abs(inst_scm) + 1e-18
                for t in tdoas_segment:
                    t_ = np.pad(t[:len(sigs)-1], (1, 0))
                    steer = np.exp(-1j * 2 * np.pi * k[:, None] / fft_size * t_)
                    ref_scm = np.einsum('fc, fd -> fcd', steer, steer.conj())
                    sim = correlation_matrix_distance(ref_scm, inst_scm)
                    mask = sim < th
                    all_masks.append(mask)
                masks = np.asarray(all_masks).astype(np.float64)

                for i in range(len(masks)):
                    for j in range(i+1, len(masks)):
                        t_i = np.pad(tdoas_segment[i][:len(sigs)-1], (1, 0))
                        t_j = np.pad(tdoas_segment[j][:len(sigs)-1], (1, 0))
                        steer_i = np.exp(-1j * 2 * np.pi * k[:, None] / fft_size * t_i)
                        steer_j = np.exp(-1j * 2 * np.pi * k[:, None] / fft_size * t_j)
                        ref_scm_i = np.einsum('fc, fd -> fcd', steer_i, steer_i.conj())
                        ref_scm_j = np.einsum('fc, fd -> fcd', steer_j, steer_j.conj())
                        decision_mask = correlation_matrix_distance(ref_scm_i, inst_scm) <= correlation_matrix_distance(ref_scm_j, inst_scm)
                        both_mask = masks[i] * masks[j]
                        both_mask = both_mask.astype(bool)
                        #both_mask[100:] = 0
                        #both_mask[abs(correlation_matrix_distance(ref_scm_i, scms)  -correlation_matrix_distance(ref_scm_j, scms)) <= .01] = 0
                        masks[i, both_mask] = decision_mask[both_mask]
                        masks[j, both_mask] = 1-decision_mask[both_mask]
                masks *= dominant


                seg_acitivities = []#np.zeros(masks.shape[:2])
                masks_reduced = []
                tdoas_reduced = []
                phantom = False
                for s, mask in enumerate(masks):
                    if verbose:
                        print(s, np.round(tdoas_segment[s], 2))
                        plt.imshow(mask.T, interpolation='nearest', aspect='auto', origin='lower')
                        plt.show()
                    act = np.mean(mask[:, k_min:k_max], -1)  > act_th

                    act = Kernel1D(erosion_len_beam, kernel=np.min)(
                        Kernel1D(dilation_len_beam, kernel=np.max)(act)
                    )
                    #act = Kernel1D(21, kernel=np.max)(act)
                    if np.sum(act) >= min_len:
                        masks_reduced.append(mask)
                        seg_acitivities.append(act)
                        tdoas_reduced.append(tdoas_segment[s])
                    else:
                        if s == 0:
                            phantom = True
                            #break
                    if verbose:
                        plt.plot(np.mean(mask[:, k_min:k_max], -1))
                        plt.plot(act_th * np.ones_like(act))
                        plt.plot(act)
                        plt.grid()
                        plt.show()
                if phantom:
                    continue
                masks = np.asarray(masks_reduced)
                seg_acitivities = np.asarray(seg_acitivities)
                '''plt.plot(seg_acitivities[0])
                plt.grid()
                plt.show()'''
                if verbose:
                    print('MM')
                trainer = CACGMMTrainer()
                permutation_aligner = DHTVPermutationAlignment(
                    stft_size=fft_size,
                    segment_start=100, segment_width=100, segment_shift=20,
                    main_iterations=20, sub_iterations=2,
                    similarity_metric='cos',
                )
                weight_constant_axis = -3
                input_mm = rearrange(sigs_stft, 'd t f -> f t d')
                masks *= seg_acitivities[..., None].astype(bool)
                init_masks = np.concatenate((masks, 1 - dominant[None]))
                init_masks = init_masks.astype(np.float64)
                max_val = .8
                init_masks[init_masks != 0] = max_val
                init_masks[init_masks == 0] = (1 - max_val) / (len(init_masks) - 1)
                init_masks = rearrange(init_masks, 'd t f -> f d t')
                source_activity_mask = np.concatenate(
                    (seg_acitivities, np.ones((1, seg_acitivities.shape[1]))),
                    axis=0
                )
                source_activity_mask = np.tile(
                    source_activity_mask[:, None], (1, 1024 // 2 + 1, 1)
                )

                cacgmm = trainer.fit(
                    input_mm,
                    initialization=init_masks,
                    weight_constant_axis=weight_constant_axis,
                    iterations=10,
                    inline_permutation_aligner=permutation_aligner
                )
                masks = cacgmm.predict(input_mm)
                masks = rearrange(masks, 'f s t -> s t f')

                seg_acitivities = []#np.zeros(masks.shape[:2])
                masks_reduced = []
                phantom = False
                for s, mask in enumerate(masks[:-1]):
                    act = np.mean(mask[:, k_min:k_max], -1)  > act_th_mm

                    act = Kernel1D(erosion_len_beam, kernel=np.min)(
                        Kernel1D(dilation_len_beam, kernel=np.max)(act)
                    )
                    if verbose:
                        plt.imshow(mask.T, interpolation='nearest', aspect='auto', origin='lower')
                        plt.show()
                        plt.plot(np.mean(mask[:, k_min:k_max], -1 ))
                        plt.plot(act_th_mm * np.ones_like(act))
                        plt.plot(act)
                        plt.grid()
                        plt.show()
                    if np.sum(act) < min_len:
                        if s == 0:
                            phantom = True
                        continue
                    act = Kernel1D(31, kernel=np.max)(act)
                    mask *= act[:, None]
                    masks_reduced.append(mask)
                    seg_acitivities.append(act)

                if phantom:
                    continue
                masks = np.asarray(masks_reduced)
                seg_acitivities = np.asarray(seg_acitivities)
                sig_segs, seg_onsets = time_varying_mvdr(sigs_stft, rearrange(masks, 's t f -> s f t'), seg_acitivities.astype(bool), wpe=False)
                for s, sig  in enumerate(sig_segs):
                    embedding =  embed_extractor(sig)
                    embeddings_mm.append(embedding)
                    seg_boundaries_mm.append((onset * 256 + seg_onsets[s], onset * 256 + seg_onsets[s] + len(sig)))
                    if verbose:
                        plt.plot(sig)
                        plt.grid()
                        plt.show()
                        pb.io.play(sig)
                ##################################################
                tdoas_segment = tdoas_reduced

                all_masks = []
                for t in tdoas_segment:
                    t_ = np.pad(t[:len(sigs)-1], (1, 0))
                    steer = np.exp(-1j * 2 * np.pi * k[:, None] / fft_size * t_)
                    ref_scm = np.einsum('fc, fd -> fcd', steer, steer.conj())
                    sim = correlation_matrix_distance(ref_scm, inst_scm)
                    mask = sim < th
                    all_masks.append(mask)
                masks = np.asarray(all_masks).astype(np.float64)

                for i in range(len(masks)):
                    for j in range(i+1, len(masks)):
                        t_i = np.pad(tdoas_segment[i][:len(sigs)-1], (1, 0))
                        t_j = np.pad(tdoas_segment[j][:len(sigs)-1], (1, 0))
                        steer_i = np.exp(-1j * 2 * np.pi * k[:, None] / fft_size * t_i)
                        steer_j = np.exp(-1j * 2 * np.pi * k[:, None] / fft_size * t_j)
                        ref_scm_i = np.einsum('fc, fd -> fcd', steer_i, steer_i.conj())
                        ref_scm_j = np.einsum('fc, fd -> fcd', steer_j, steer_j.conj())
                        decision_mask = correlation_matrix_distance(ref_scm_i, inst_scm) <= correlation_matrix_distance(ref_scm_j, inst_scm)
                        both_mask = masks[i] * masks[j]
                        both_mask = both_mask.astype(bool)
                        #both_mask[100:] = 0
                        #both_mask[abs(correlation_matrix_distance(ref_scm_i, scms)  -correlation_matrix_distance(ref_scm_j, scms)) <= .01] = 0
                        masks[i, both_mask] = decision_mask[both_mask]
                        masks[j, both_mask] = 1-decision_mask[both_mask]
                masks *= dominant


                seg_acitivities = []#np.zeros(masks.shape[:2])
                masks_reduced = []
                tdoas_reduced = []
                phantom = False
                for s, mask in enumerate(masks):
                    if verbose:
                        print(s, np.round(tdoas_segment[s], 2))
                        plt.imshow(mask.T, interpolation='nearest', aspect='auto', origin='lower')
                        plt.show()
                    act = np.mean(mask[:, k_min:k_max], -1)  > act_th2

                    act = Kernel1D(erosion_len_beam, kernel=np.min)(
                        Kernel1D(dilation_len_beam, kernel=np.max)(act)
                    )
                    act = Kernel1D(31, kernel=np.max)(act)
                    if np.sum(act) >= min_len:
                        masks_reduced.append(mask)
                        seg_acitivities.append(act)
                        tdoas_reduced.append(tdoas_segment[s])
                    else:
                        if s == 0:
                            phantom = True
                            #break
                    if verbose:
                        plt.plot(np.mean(mask[:, k_min:k_max], -1))
                        plt.plot(act_th * np.ones_like(act))
                        plt.plot(act)
                        plt.grid()
                        plt.show()
                if phantom:
                    continue
                masks = np.asarray(masks_reduced)
                seg_acitivities = np.asarray(seg_acitivities)
                sig_segs, seg_onsets = time_varying_mvdr(sigs_stft, rearrange(masks, 's t f -> s f t'), seg_acitivities.astype(bool), wpe=False)
                for s, sig  in enumerate(sig_segs):
                    embedding =  embed_extractor(sig)
                    embeddings.append(embedding)
                    seg_boundaries.append((onset * 256 + seg_onsets[s], onset * 256 + seg_onsets[s] + len(sig)))

            embeddings_red = []
            activities_red = []
            embeddings_short = []
            activities_short = []
            for i, e in enumerate(embeddings):
                onset, offset = seg_boundaries[i]
                if offset - onset > 16000:
                    embeddings_red.append(e)
                    activities_red.append((onset, offset))
                else:
                    embeddings_short.append(e)
                    activities_short.append((onset, offset))
            labels  = HDBSCAN(
                min_cluster_size=3, min_samples=3, cluster_selection_epsilon=0.,
                max_cluster_size=None, metric='cosine'
            ).fit_predict(embeddings_red)

            num_spk = np.max(labels) + 1
            est_activities = np.zeros((num_spk, sigs.shape[-1]))

            for spk_id in range(num_spk):
                for i in range(len(activities_red)):
                    if labels[i] == spk_id:
                        on, off = activities_red[i]
                        est_activities[spk_id, on:off] = 1
            for i in range(len(activities_red)):
                if labels[i] == -1:
                    on, off = activities_red[i]
                    dists = [cos_dist(embeddings_red[i], spk_emdeb) if labels[
                                                                           s] != -1 else 10
                             for s, spk_emdeb in enumerate(embeddings_red)]
                    closest_spk = labels[np.argmin(dists)]
                    est_activities[closest_spk, on:off] = 1

            embeddings_red = []
            activities_red = []

            embeddings_short = []
            activities_short = []
            for i, e in enumerate(embeddings_mm):
                onset, offset = seg_boundaries_mm[i]
                if offset - onset > 16000:
                    embeddings_red.append(e)
                    activities_red.append((onset, offset))
                else:
                    embeddings_short.append(e)
                    activities_short.append((onset, offset))
            labels  = HDBSCAN(
                min_cluster_size=3, min_samples=3, cluster_selection_epsilon=0.,
                max_cluster_size=None, metric='cosine'
            ).fit_predict(embeddings_red)

            num_spk = np.max(labels) + 1

            est_activities_mm = np.zeros((num_spk, sigs.shape[-1]))

            for spk_id in range(num_spk):
                for i in range(len(activities_red)):
                    if labels[i] == spk_id:
                        on, off = activities_red[i]
                        est_activities_mm[spk_id, on:off] = 1
            for i in range(len(activities_red)):
                if labels[i] == -1:
                    on, off = activities_red[i]
                    dists = [cos_dist(embeddings_red[i], spk_emdeb) if labels[s] != -1 else 10 for s, spk_emdeb in enumerate(embeddings_red) ]
                    closest_spk = labels[np.argmin(dists)]
                    est_activities_mm[closest_spk, on:off] = 1

            ###################################################################
            spatial[session_name] = {spk: pb.array.interval.ArrayInterval(act.astype(bool))
                                                   for spk, act in enumerate(est_activities_spatial)}
            est_diarization_mm[session_name] = {spk: pb.array.interval.ArrayInterval(act.astype(bool))
                                                   for spk, act in enumerate(est_activities_mm)}
            est_diarization_wo_mm[session_name] = {spk: pb.array.interval.ArrayInterval(act.astype(bool))
                                                   for spk, act in enumerate(est_activities)}

            del sigs_stft_complete, sigs, voice_activity, sigs_stft, frame_wise_voice_activity, dominant, eig_val_mem, sigs_stft_, scms, dominant_complete, inst_scm
        spatial = spatial.gather()
        est_diarization_mm = est_diarization_mm.gather()
        est_diarization_wo_mm = est_diarization_wo_mm.gather()
        Path(experiment_dir).mkdir(exist_ok=True, parents=True)
        if dlp_mpi.IS_MASTER:
            pb.array.interval.rttm.to_rttm(spatial, Path(experiment_dir) / f'diarization_estimates_spattial_{dset}.rttm')
            pb.array.interval.rttm.to_rttm(est_diarization_mm, Path(experiment_dir) / f'diarization_estimates_mm_{dset}.rttm')
            pb.array.interval.rttm.to_rttm(est_diarization_wo_mm, Path(experiment_dir) / f'diarization_estimates_wo_mm_{dset}.rttm')



@ex.main
def main(_config, _run):
    '''experiment_dir = Path(_config['experiment_dir'])
    pt.io.dump_config(
        copy.deepcopy(_config),
        Path(experiment_dir) / 'config.json'
    )
    pb.io.dump_json(_config, Path(_config['experiment_dir']) / 'config.json')'''
    spatio_spectral_pipeline()


if __name__ == '__main__':
    ex.run_commandline()
