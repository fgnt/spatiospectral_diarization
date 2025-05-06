from einops import rearrange

import numpy as np
from scipy.signal import fftconvolve
import paderbox as pb
from paderbox.transform.module_stft import stft_frame_index_to_sample_index
from pb_bss.math.solve import stable_solve
from pb_bss.distribution.cacgmm import CACGMMTrainer
from pb_bss.permutation_alignment import DHTVPermutationAlignment, OraclePermutationAlignment
from pb_bss.extraction.beamformer import get_power_spectral_density_matrix
from pb_bss.extraction.beamformer import get_mvdr_vector_souden
from pb_bss.extraction.beamformer import get_mvdr_vector
from pb_bss.extraction.beamformer import blind_analytic_normalization
from pb_bss.extraction.beamformer_wrapper import get_gev_rank_one_estimate
from tcrl.utils import Kernel1D
from nara_wpe.wpe import wpe_v8

from spatial_diarization.separation.utils import (
    get_sdrs,
    get_interference_segments
)


def time_freq_init(
    sigs_stft, activities, tdoas, fft_size=1024, max_val=.8,
    kernel_size_scm_smoothing=3, eig_val_ratio_th=.9, eig_val_th=1e-3
):
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
    eig_vals, _ = np.linalg.eigh(scms)
    dominance = 1 - eig_vals[..., -2] / eig_vals[..., -1]
    dominant = (dominance >= eig_val_ratio_th)
    dominant *= (eig_vals[..., -1] > eig_val_th)

    energy = []
    steering_vects = []
    for i, tdoa_vec in enumerate(tdoas):
        tdoa_vec = np.pad(tdoa_vec, (1, 0))[:len(sigs_stft)]
        steering_vect = np.exp(
            -2j * np.pi *np.arange(fft_size // 2 + 1)
            / fft_size * tdoa_vec[..., None]
        )
        steering_vects.append(steering_vect)
    for i, steering_vect in enumerate(steering_vects):
        noise_scm = 0
        for c in range(len(steering_vects)):
            if c == i:
                continue
            noise_scm += np.einsum(
                'c f, d f -> f c d', steering_vects[c], steering_vects[c].conj()
            )
        noise_scm += 1e-9 * np.eye(len(sigs_stft))[None]
        bf_vec = get_mvdr_vector(steering_vects[i].T, noise_scm)
        bf_output = np.einsum('t f c, c t f ->  f t', bf_vec.conj()[None], sigs_stft)
        energy.append(abs(bf_output) ** 2)
    energy = np.asarray(energy)
    energy *= activities[:, None]
    init_masks = np.zeros_like(energy)
    for t in range(energy.shape[1]):
        for f in range(energy.shape[2]):
            idx = np.argmax(energy[:, t, f])
            init_masks[idx, t, f] = 1
    init_masks *= dominant[None]
    init_masks = np.concatenate((init_masks, 1 - np.sum(init_masks, 0, keepdims=True)), 0)
    #activities = np.pad(activities.astype(np.float64), ((0, 1), (0, 0)), constant_values=1)
    #activities = np.tile(activities[:, None], (1, fft_size // 2 + 1, 1))
    #activities *= (1 - max_val) / np.sum(activities, 0, keepdims=True)
    #activities[init_masks != 0] = max_val
    #init_masks = rearrange(activities, 'd f t -> f d t')
    init_masks[init_masks != 0] = max_val
    init_masks[init_masks == 0] = (1 - max_val) / (len(init_masks) - 1)
    init_masks = rearrange(init_masks, 'd f t -> f d t')
    return init_masks


def time_init(activities, fft_size=1024, max_val=.8):
    init_masks = np.concatenate(
        (activities, np.zeros((1, activities.shape[-1]))), axis=0
    )
    init_masks /= np.sum(init_masks, 0) + np.finfo(np.float64).tiny
    init_masks[init_masks > 1e-3] *= max_val
    init_masks[-1] = 1 - max_val
    init_masks = np.tile(
        init_masks[None], (fft_size // 2 + 1, 1, 1)
    )
    return init_masks


def estimate_masks(
        sigs_stft, spk_id, frame_activities, tdoas, guided_iter, non_guided_iter,
        onset, offset, frame_size_enh, fft_size, verbose=False, tf_init=True,
        time_dependent_prior=True
):
    spk_mask = np.sum(frame_activities[:, onset:offset], -1) > 0
    spk_mask[spk_id] = 0
    others = frame_activities[spk_mask, onset:offset].copy()
    if np.sum(spk_mask):
        source_activity_mask = np.concatenate(
            (frame_activities[spk_id, None, onset:offset], others), axis=0
        )
        tdoas_seg = np.concatenate((tdoas[spk_id, None], tdoas[spk_mask]) , axis=0)
    else:
        source_activity_mask = \
            frame_activities[spk_id, None, onset:offset].copy()
        tdoas_seg = tdoas[spk_id, None].copy()
    source_activity_mask = source_activity_mask.astype(bool)
    if tf_init:
        init = time_freq_init(
            sigs_stft, source_activity_mask, tdoas_seg, fft_size=fft_size
        )
    else:
        init = time_init(source_activity_mask)
    if guided_iter == 0 and non_guided_iter == 0:
        masks = rearrange(init > .5, 'f s t -> s f t').astype(np.float64)
        return masks
    source_activity_mask = np.concatenate(
        (source_activity_mask, np.ones((1, np.minimum(offset, frame_activities.shape[-1])-onset))),
        axis=0
    )
    source_activity_mask = np.tile(
        source_activity_mask[:, None], (1, fft_size // 2 + 1, 1)
    )
    source_activity_mask = source_activity_mask.astype(bool)
    trainer = CACGMMTrainer()
    if time_dependent_prior:
        permutation_aligner = DHTVPermutationAlignment(
            stft_size=fft_size,
            segment_start=100, segment_width=100, segment_shift=20,
            main_iterations=20, sub_iterations=2,
            similarity_metric='cos',
        )
        weight_constant_axis = -3
    else:
        permutation_aligner = None
        weight_constant_axis = -1
    input_mm = rearrange(sigs_stft, 'd t f -> f t d')
    if guided_iter > 0:
        cacgmm = trainer.fit(
            input_mm,
            initialization=init,
            weight_constant_axis=weight_constant_axis,
            iterations=guided_iter,
            source_activity_mask=rearrange(source_activity_mask, 'd f t -> f d t')
        )
        masks = cacgmm.predict(input_mm)
        init = cacgmm
    if non_guided_iter > 0:
        trainer = CACGMMTrainer()
        cacgmm = trainer.fit(
            input_mm,
            initialization=init,
            weight_constant_axis=weight_constant_axis,
            iterations=non_guided_iter,
            inline_permutation_aligner=permutation_aligner,
        )
        masks = cacgmm.predict(input_mm)

    global_pa = OraclePermutationAlignment()
    global_pa_est = rearrange(masks, 'f s t -> s (t f)')
    global_pa_reference = rearrange(source_activity_mask, 's f t  -> s (t f)')
    global_permutation = global_pa.calculate_mapping(global_pa_est, global_pa_reference)
    masks = rearrange(masks, 'f s t -> s f t')
    masks = masks[global_permutation]
    return masks


def switching_gss(
        spk_id, sigs, frame_activities, tdoas, wpe=True, frame_size=1024,
        fft_size=1024, context=5, guided_iter=5, non_guided_iter=5,
        act_th=.2, erosion_len_beam=63, dilation_len_beam=125,
        min_len=32, verbose=False, plt_masks=False, tf_init=True,
        time_dependent_prior=True, masking_min=1., ban=False, rank1=False
):
    frame_shift_enh = frame_size // 4
    context = int(np.round((context * 16000) / frame_shift_enh))
    eps = 1e-18
    sig_segments = []
    segment_onsets = []
    num_samples_segments = []

    if context % 2 == 0:
        spk_act = Kernel1D(context + 1, kernel=np.min)(
            Kernel1D(context + 1, kernel=np.max)(frame_activities[spk_id])
        )
    else:
        spk_act = Kernel1D(context, kernel=np.min)(
            Kernel1D(context, kernel=np.max)(frame_activities[spk_id])
        )

    activity_intervals = \
        pb.array.interval.ArrayInterval(spk_act).normalized_intervals
    for seg_id, (onset_, offset_) in enumerate(activity_intervals):
        if offset_ - onset_ < min_len:
            continue
        onset = np.maximum(onset_ - context, 0)
        offset = offset_ + context
        time_onset = stft_frame_index_to_sample_index(
            onset, frame_size, frame_size // 4, pad=False, fading=False, mode='first'
        )
        time_offset = stft_frame_index_to_sample_index(
            offset, frame_size, frame_size // 4, pad=False, fading=False, mode='last'
        )
        sigs_stft_beam = pb.transform.stft(
            sigs[:, time_onset:time_offset], size= fft_size, shift=frame_size//4,window_length=frame_size,
            pad=False, fading=False
        )
        if wpe:
            sigs_stft_beam = wpe_v8(
                rearrange(sigs_stft_beam, 'd t f -> f d t')
            )
            sigs_stft_beam = rearrange(sigs_stft_beam, 'f d t -> d t f')

        masks = estimate_masks(
            sigs_stft_beam, spk_id, frame_activities, tdoas, guided_iter,
            non_guided_iter, onset, offset, frame_size, fft_size,
            verbose=plt_masks, tf_init=tf_init,
            time_dependent_prior=time_dependent_prior
        )

        activities = []
        for i, m in enumerate(masks):
            activity = np.mean(m, 0) >= act_th
            activity = Kernel1D(erosion_len_beam, kernel=np.min)(
                Kernel1D(dilation_len_beam, kernel=np.max)(activity)
            )
            activities.append(activity)
        activities = np.asarray(activities)
        activities = activities[:-1]

        target_act = activities[0]
        activity_intervals = \
            pb.array.interval.ArrayInterval(target_act).normalized_intervals

        for on, off in activity_intervals:
            if off - on < min_len:
                continue
            scm_target = get_power_spectral_density_matrix(
                rearrange(sigs_stft_beam[:, on:off], 'c t f -> f c t') * masks[0, :, None,  on:off],
                normalize=False
            )
            scm_target /= off - on
            segment_info = get_interference_segments(activities, on, off, 8)
            interference_segments = []
            for s, (on_, of_, concurrent) in enumerate(segment_info):
                interference_mask = 1 - masks[0, :, None,  on_:of_]
                interference_scm = get_power_spectral_density_matrix(
                    rearrange(sigs_stft_beam[:, on_:of_], 'c t f -> f c t') * interference_mask,
                    normalize=False
                )
                interference_scm /= of_ - on_
                interference_scm += \
                    1e-8 * np.eye(sigs.shape[0])[None]
                if rank1:
                    scm_target_ = \
                        get_gev_rank_one_estimate(scm_target, interference_scm)
                else:
                    scm_target_ = scm_target.copy()
                interference_segments.append(
                    (on_, of_, interference_scm, scm_target_)
                )
            all_sdrs = []
            for (on_, of_, interference_scm, scm_target) in interference_segments:
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
            for (on_, of_, interference_scm, scm_target) in interference_segments:
                bf_vec = get_mvdr_vector_souden(
                    scm_target, interference_scm, ref_channel=ref_ch
                )
                if ban:
                    bf_vec = \
                        blind_analytic_normalization(bf_vec, interference_scm)
                for l in range(on_, of_):
                    bf_output[l-on] = \
                        np.einsum('fc, cf-> f', np.conj(bf_vec), sigs_stft_beam[:, l])
            if masking_min < 1.:
                bf_output *= np.maximum(masks[0, :, on:off].T, masking_min)
            enh_sig = pb.transform.istft(
                bf_output, size=frame_size, shift=frame_size//4,
                window_length=frame_size, fading=False
            )
            sig_segments.append(enh_sig)
            global_onset = \
                pb.transform.module_stft.stft_frame_index_to_sample_index(
                    onset, window_length=frame_size, shift=frame_shift_enh,
                    pad=False, fading=False, mode='first'
                )
            global_onset += \
                pb.transform.module_stft.stft_frame_index_to_sample_index(
                    on, window_length=frame_size, shift=frame_size//4,
                    pad=False, fading=False, mode='first'
                )
            segment_onsets.append(global_onset)
            num_samples_segments.append(len(enh_sig))
    return sig_segments, segment_onsets, num_samples_segments


def gss(
        spk_id, sigs, frame_activities, tdoas, wpe=True, frame_size=1024,
        fft_size=1024, context=5, guided_iter=5, non_guided_iter=5,
        act_th=.2, erosion_len_beam=63, dilation_len_beam=125,
        min_len=32, verbose=False, plt_masks=False, tf_init=True
):
    frame_shift_enh = frame_size // 4
    context = int(np.round((context * 16000) / frame_shift_enh))
    eps = 1e-18
    sig_segments = []
    segment_onsets = []
    num_samples_segments = []

    if context % 2 == 0:
        spk_act = Kernel1D(context + 1, kernel=np.min)(
            Kernel1D(context + 1, kernel=np.max)(frame_activities[spk_id])
        )
    else:
        spk_act = Kernel1D(context, kernel=np.min)(
            Kernel1D(context, kernel=np.max)(frame_activities[spk_id])
        )

    activity_intervals = \
        pb.array.interval.ArrayInterval(spk_act).normalized_intervals
    for seg_id, (onset_, offset_) in enumerate(activity_intervals):
        if offset_ - onset_ < min_len:
            continue
        onset = np.maximum(onset_ - context, 0)
        offset = offset_ + context
        time_onset = stft_frame_index_to_sample_index(
            onset, frame_size, frame_size // 4, pad=False, fading=False, mode='first'
        )
        time_offset = stft_frame_index_to_sample_index(
            offset, frame_size, frame_size // 4, pad=False, fading=False, mode='last'
        )
        sigs_stft_beam = pb.transform.stft(
            sigs[:, time_onset:time_offset], size= fft_size, shift=frame_size//4,window_length=frame_size,
            pad=False, fading=False
        )
        if wpe:
            sigs_stft_beam = wpe_v8(
                rearrange(sigs_stft_beam, 'd t f -> f d t')
            )
            sigs_stft_beam = rearrange(sigs_stft_beam, 'f d t -> d t f')

        masks = estimate_masks(
            sigs_stft_beam, spk_id, frame_activities, tdoas, guided_iter,
            non_guided_iter, onset, offset, frame_size, fft_size,
            verbose=plt_masks, tf_init=tf_init
        )

        activities = []
        for i, m in enumerate(masks):
            activity = np.mean(m, 0) >= act_th
            activity = Kernel1D(erosion_len_beam, kernel=np.min)(
                Kernel1D(dilation_len_beam, kernel=np.max)(activity)
            )
            activities.append(activity)
        activities = np.asarray(activities)
        activities = activities[:-1]

        target_act = activities[0]
        activity_intervals = \
            pb.array.interval.ArrayInterval(target_act).normalized_intervals

        for on, off in activity_intervals:
            if off - on < min_len:
                continue
            scm_target = get_power_spectral_density_matrix(
                rearrange(sigs_stft_beam[:, on:off], 'c t f -> f c t') * masks[0, :, None,  on:off],
                normalize=False
            )
            scm_target /= off - on

            interference_scm = get_power_spectral_density_matrix(
                rearrange(sigs_stft_beam[:, on:off], 'c t f -> f c t') * (1 - masks[0, :, None,  on:off]),
                normalize=False
            )
            interference_scm /= off - on
            interference_scm += \
                1e-8 * np.eye(sigs.shape[0])[None]
            phi = stable_solve(interference_scm, scm_target)
            lambda_ = np.trace(phi, axis1=-1, axis2=-2)[..., None, None]
            if eps is None:
                eps = np.finfo(lambda_.dtype).tiny
            w_mat = phi / np.maximum(lambda_.real, eps)
            sdrs_segment = get_sdrs(w_mat, scm_target, interference_scm)
            ref_ch = np.argmax(sdrs_segment.real)

            bf_output = np.zeros(
                (off - on, sigs_stft_beam.shape[-1]), np.complex128
            )
            bf_vec = get_mvdr_vector_souden(
                scm_target, interference_scm, ref_channel=ref_ch
            )
            for l in range(on, off):
                bf_output[l-on] = \
                    np.einsum('fc, cf-> f', np.conj(bf_vec), sigs_stft_beam[:, l])
            enh_sig = pb.transform.istft(
                bf_output, size=frame_size, shift=frame_size//4,
                window_length=frame_size, fading=False
            )
            sig_segments.append(enh_sig)
            global_onset = \
                pb.transform.module_stft.stft_frame_index_to_sample_index(
                    onset, window_length=frame_size, shift=frame_shift_enh,
                    pad=False, fading=False, mode='first'
                )
            global_onset += \
                pb.transform.module_stft.stft_frame_index_to_sample_index(
                    on, window_length=frame_size, shift=frame_size//4,
                    pad=False, fading=False, mode='first'
                )
            segment_onsets.append(global_onset)
            num_samples_segments.append(len(enh_sig))
    return sig_segments, segment_onsets, num_samples_segments


"""from einops import rearrange

import numpy as np
import paderbox as pb
from paderbox.transform.module_stft import stft_frame_index_to_sample_index
from pb_bss.math.solve import stable_solve
from pb_bss.distribution.cacgmm import CACGMMTrainer
from pb_bss.permutation_alignment import DHTVPermutationAlignment, OraclePermutationAlignment
from pb_bss.extraction.beamformer import get_power_spectral_density_matrix
from pb_bss.extraction.beamformer import get_mvdr_vector_souden
from pb_bss_cbj.distribution.cacgmm_v2 import CACGMMTrainer_v2
from mmmeeting.utils import Kernel1D
from nara_wpe.wpe import wpe_v8

import matplotlib.pyplot as plt
from spatial_diarization.mask_estimation.utils import plot_masks


def get_sdrs(w_mat, target_psd_matrix, noise_psd_matrix, eps=None):
    if w_mat.ndim != 3:
        raise ValueError(
            'Estimating the ref_channel expects currently that the input '
            'has 3 ndims (frequency x sensors x sensors). '
            'Considering an independent dim in the SNR estimate is not '
            'unique.'
        )
    if eps is None:
        eps = np.finfo(w_mat.dtype).tiny
    sdrs = np.einsum(
        '...FdR,...FdD,...FDR->...R', w_mat.conj(), target_psd_matrix, w_mat
    ) / np.maximum(np.einsum(
        '...FdR,...FdD,...FDR->...R', w_mat.conj(), noise_psd_matrix, w_mat
    ), eps)
    # Raises an exception when np.inf and/or np.NaN was in target_psd_matrix
    # or noise_psd_matrix
    assert np.all(np.isfinite(sdrs)), sdrs
    return sdrs.real


def soft_mask(activities, num_classes, maximum=.9, fft_size=1024):
    activities = activities.astype(np.float32)
    activities = [Kernel1D(63, kernel=np.amin)(Kernel1D(63, kernel=np.amax)(a))
                  for a in activities]
    while len(activities) < num_classes - 1:
        activities.append(np.ones(len(activities[0])))
    activities.append(np.zeros(len(activities[0])))
    initialization = np.asarray(activities)
    initialization /= np.maximum(
        np.sum(activities, axis=0, keepdims=True), np.finfo(np.float64).eps
    )
    initialization[initialization > 0] *= maximum
    remainder = 1 - np.sum(initialization, 0)
    remainder /= \
        np.maximum(np.sum(initialization == 0, 0), np.finfo(np.float64).eps)
    for s in range(len(initialization)):
        time_mask = initialization[s] == 0
        initialization[s, time_mask] = remainder[time_mask]
    initialization = np.tile(initialization[None], (fft_size // 2 + 1, 1, 1))
    return initialization


def fit_cacagmm(
    input_mm, initialization, iterations, weight_constant_axis=-3,
    source_activity_mask=None, inline_permutation_aligner=None
):
    if weight_constant_axis == -1:
        inline_permutation_aligner = None
    trainer = CACGMMTrainer_v2(
        y=rearrange(input_mm, 'f t d -> f d t'),
        initialization=initialization,
        iterations=iterations,
        weight_constant_axis=weight_constant_axis,
        inline_permutation_aligner=None,
        loopy=True,
    )
    for i in range(trainer.iterations):
        if trainer.model is not None:
            trainer.e_step()
            if trainer.inline_permutation_aligner is not None:
                trainer.permutation_align()
        else:
            pass

        if source_activity_mask is not None:
            affiliation = trainer.affiliation.copy()
            affiliation *= source_activity_mask
            trainer.affiliation = affiliation
        trainer.m_step()
    trainer.e_step()

    if weight_constant_axis in [-3, (-3)]:
        model = trainer.get_model()
    affiliation = trainer.affiliation
    if inline_permutation_aligner is not None:
        mapping = inline_permutation_aligner.calculate_mapping(
            rearrange(affiliation, 'f k t ->k f t'))

        masks = rearrange(inline_permutation_aligner.apply_mapping(
            rearrange(affiliation, 'f k t ->k f t'), mapping
        ), 'k f t -> f k t')
        return masks
    return affiliation


def estimate_masks(
        sigs_stft, spk_id, frame_activities, guided_iter, non_guided_iter,
        onset, offset, frame_size_enh, fft_size, verbose=False
):
    spk_mask = np.sum(frame_activities[:, onset:offset], -1) > 0
    spk_mask[spk_id] = 0
    others = frame_activities[spk_mask, onset:offset]
    if np.sum(spk_mask):
        source_activity_mask = np.concatenate(
            (frame_activities[spk_id, None, onset:offset], others), axis=0
        )
    else:
        source_activity_mask = np.concatenate(
            (frame_activities[spk_id, None, onset:offset]), axis=0
        )
    source_activity_mask = source_activity_mask.astype(bool)
    init = soft_mask(source_activity_mask, len(source_activity_mask),
                     fft_size=fft_size)
    source_activity_mask = np.concatenate(
        (source_activity_mask, np.ones((1, np.minimum(offset, frame_activities.shape[-1])-onset))),
        axis=0
    )
    source_activity_mask = np.tile(
        source_activity_mask[:, None], (1, fft_size // 2 + 1, 1)
    )
    source_activity_mask = source_activity_mask.astype(bool)
    if verbose:
        plot_masks(source_activity_mask)
        plt.show()
    trainer = CACGMMTrainer()
    permutation_aligner = DHTVPermutationAlignment(
        stft_size=fft_size,
        segment_start=100, segment_width=100, segment_shift=20,
        main_iterations=20, sub_iterations=2,
        similarity_metric='cos',
    )
    input_mm = rearrange(sigs_stft, 'd t f -> f t d')

    cacgmm = trainer.fit(
        input_mm,
        initialization=init,
        weight_constant_axis=-3,
        iterations=guided_iter,
        source_activity_mask=rearrange(source_activity_mask, 'd f t -> f d t')
    )
    masks = cacgmm.predict(input_mm)
    '''masks = fit_cacagmm(
        input_mm,
        initialization=init,
        weight_constant_axis=-3,
        iterations=guided_iter,
        source_activity_mask=rearrange(source_activity_mask, 'd f t -> f d t')
    )'''
    if non_guided_iter > 0:
        trainer = CACGMMTrainer()
        cacgmm = trainer.fit(
            input_mm,
            initialization=masks,
            weight_constant_axis=-3,
            iterations=non_guided_iter,
            inline_permutation_aligner=permutation_aligner,
        )
        masks = cacgmm.predict(input_mm)
        '''masks = fit_cacagmm(
            input_mm,
            initialization=masks,
            weight_constant_axis=-3,
            iterations=non_guided_iter,
            inline_permutation_aligner=permutation_aligner
        )'''
    if False:#verbose:
        plot_masks(rearrange(masks, 'f s t -> s f t'))
        plt.show()
    global_pa = OraclePermutationAlignment()
    global_pa_est = rearrange(masks, 'f s t -> s (t f)')
    global_pa_reference = rearrange(source_activity_mask, 's f t  -> s (t f)')
    global_permutation = global_pa.calculate_mapping(global_pa_est, global_pa_reference)
    masks = rearrange(masks, 'f s t -> s f t')
    masks = masks[global_permutation]
    if verbose:
        plot_masks(masks)
        plt.show()
    return masks


def get_interference_segments(activities, onset, offset, min_len=16):
    segments = []
    for other_spk in range(1, len(activities)):
        intersectons = pb.array.interval.ArrayInterval(
            activities[0, onset:offset] * activities[other_spk, onset:offset]
        )
        intersecton_intervals = intersectons.normalized_intervals
        for (intersect_onset, intersect_offset) in intersecton_intervals:
            segments.append(
                (other_spk, onset + intersect_onset, onset + intersect_offset)
            )
    borders = [segment[1] for segment in segments] + [segment[-1]
               for segment in segments]
    borders += [onset, offset]
    borders = np.sort(list(set(borders)))
    borders_ = [borders[0], ]
    for i in range(1, len(borders)):
        if borders[i] - borders[i-1] < min_len:
            if i == 1:
                continue
            else:
                borders_.remove(borders_[-1])
                borders_.append((borders[i] + borders[i-1]) // 2)
        else:
            borders_.append(borders[i])
    borders = np.sort(list(set(borders_)))
    segment_info = []
    for i in range(1, len(borders)):
        concurrent = [
            other_spk for other_spk in range(1, len(activities))
            if np.sum(activities[other_spk, borders[i-1]:borders[i]]) > 0
        ]
        segment_info.append((borders[i-1], borders[i], concurrent))
    return segment_info


def switching_gss(
        spk_id, sigs, frame_activities, wpe=True, frame_size=1024, fft_size=1024,
        context=3, guided_iter=20, non_guided_iter=5,
        act_th=.2, erosion_len_beam=63, dilation_len_beam=125,
        min_len=32, verbose=False, plt_masks=False
):
    frame_shift_enh = frame_size // 4
    context = int(np.round((context * 16000) / frame_shift_enh))
    eps = 1e-18
    sig_segments = []
    segment_onsets = []
    num_samples_segments = []

    if context % 2 == 0:
        spk_act = Kernel1D(context + 1, kernel=np.min)(
            Kernel1D(context + 1, kernel=np.max)(frame_activities[spk_id])
        )
    else:
        spk_act = Kernel1D(context, kernel=np.min)(
            Kernel1D(context, kernel=np.max)(frame_activities[spk_id])
        )

    activity_intervals = \
        pb.array.interval.ArrayInterval(spk_act).normalized_intervals
    for seg_id, (onset_, offset_) in enumerate(activity_intervals):
        if offset_ - onset_ < min_len:
            continue
        onset = np.maximum(onset_ - context, 0)
        offset = offset_ + context
        time_onset = stft_frame_index_to_sample_index(
            onset, frame_size, frame_size // 4, pad=False, fading=False, mode='first'
        )
        time_offset = stft_frame_index_to_sample_index(
            offset, frame_size, frame_size // 4, pad=False, fading=False, mode='last'
        )
        sigs_stft_beam = pb.transform.stft(
            sigs[:, time_onset:time_offset], size= fft_size, shift=frame_size//4,window_length=frame_size,
            pad=False, fading=False
        )
        if wpe:
            sigs_stft_beam = wpe_v8(
                rearrange(sigs_stft_beam, 'd t f -> f d t')
            )
            sigs_stft_beam = rearrange(sigs_stft_beam, 'f d t -> d t f')

        masks = estimate_masks(
            sigs_stft_beam, spk_id, frame_activities, guided_iter,
            non_guided_iter, onset, offset, frame_size, fft_size, verbose=plt_masks
        )

        activities = []
        for i, m in enumerate(masks):
            activity = np.mean(m, 0) >= act_th
            activity = Kernel1D(erosion_len_beam, kernel=np.min)(
                Kernel1D(dilation_len_beam, kernel=np.max)(activity)
            )
            if False:#verbose:
                plt.plot(np.mean(m, 0))
                plt.plot(activity)
                plt.grid()
                plt.show()
            activities.append(activity)
        activities = np.asarray(activities)
        activities = activities[:-1]

        target_act = activities[0]
        activity_intervals = \
            pb.array.interval.ArrayInterval(target_act).normalized_intervals
        for on, off in activity_intervals:
            if off - on < min_len:
                continue
            scm_target = get_power_spectral_density_matrix(
                rearrange(sigs_stft_beam[:, on:off], 'c t f -> f c t') * masks[0, :, None,  on:off],
                normalize=False
            )
            scm_target /= off - on

            segment_info = get_interference_segments(activities, on, off, 8)
            interference_segments = []
            for s, (on_, of_, concurrent) in enumerate(segment_info):
                interference_mask = 1 - masks[0, :, None,  on_:of_]
                interference_scm = get_power_spectral_density_matrix(
                    rearrange(sigs_stft_beam[:, on_:of_], 'c t f -> f c t') * interference_mask,
                    normalize=False
                )
                interference_scm /= of_ - on_
                interference_scm += \
                    1e-8 * np.eye(sigs.shape[0])[None]
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
            if verbose:
                print(all_sdrs)
            all_sdrs = np.min(all_sdrs, 0)
            if verbose:
                print(all_sdrs)
            ref_ch = np.argmax(all_sdrs.real)
            if verbose:
                print('Ref. Ch.:', ref_ch)
            bf_output = np.zeros(
                (off - on, sigs_stft_beam.shape[-1]), np.complex128
            )
            for (on_, of_, interference_scm) in interference_segments:
                bf_vec = get_mvdr_vector_souden(
                    scm_target, interference_scm, ref_channel=ref_ch
                )
                for l in range(on_, of_):
                    bf_output[l-on] = \
                        np.einsum('fc, cf-> f', np.conj(bf_vec), sigs_stft_beam[:, l])
            enh_sig = pb.transform.istft(
                bf_output, size=frame_size, shift=frame_size//4,
                window_length=frame_size, fading=False
            )
            if verbose:
                pb.io.play(enh_sig)
            sig_segments.append(enh_sig)
            global_onset = \
                pb.transform.module_stft.stft_frame_index_to_sample_index(
                    onset, window_length=frame_size, shift=frame_shift_enh,
                    pad=False, fading=False, mode='first'
                )
            global_onset += \
                pb.transform.module_stft.stft_frame_index_to_sample_index(
                    on, window_length=frame_size, shift=frame_size//4,
                    pad=False, fading=False, mode='first'
                )
            segment_onsets.append(global_onset)
            num_samples_segments.append(len(enh_sig))
    return sig_segments, segment_onsets, num_samples_segments"""
