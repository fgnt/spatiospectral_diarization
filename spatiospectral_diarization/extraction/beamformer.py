from einops import rearrange

import numpy as np
import paderbox as pb
from pb_bss.math.solve import stable_solve
from pb_bss.extraction.beamformer import get_power_spectral_density_matrix
from pb_bss.extraction.beamformer import get_mvdr_vector_souden
from nara_wpe.wpe import wpe_v8

from spatial_diarization.separation.utils import (
    get_sdrs,
    get_interference_segments
)


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
            for l in range(on_, of_):
                bf_output[l] = \
                    np.einsum('fc, cf-> f', np.conj(bf_vec), sigs_stft_beam[:, l])
        enh_sig = pb.transform.istft(
            bf_output, size=frame_size, shift=frame_size//4,
            window_length=frame_size, fading=False
        )
        sig_segments.append(enh_sig)
        onset = \
            pb.transform.module_stft.stft_frame_index_to_sample_index(
                on, window_length=frame_size, shift=frame_size // 4,
                pad=False, fading=False, mode='first'
            )
        segment_onsets.append(onset)
    return sig_segments, segment_onsets