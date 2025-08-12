from libriwasn.mask_estimation.initialization import correlation_matrix_distance
from einops import rearrange
import numpy as np
from scipy.signal import fftconvolve
import paderbox as pb
from pb_bss.distribution.cacgmm import CACGMMTrainer
from pb_bss.permutation_alignment import DHTVPermutationAlignment
from nara_wpe.wpe import wpe_v8

def compute_dominant_mask(scms, eig_val_ratio_th):
    eig_vals, eig_vects = np.linalg.eigh(scms)
    eig_val_th = np.min(eig_vals) * 10
    dominance = 1 - eig_vals[..., -2] / eig_vals[..., -1]
    dominant = (dominance >= eig_val_ratio_th)
    dominant *= (eig_vals[..., -1] > eig_val_th)
    dominant = dominant.T
    return dominant

def compute_smoothed_scms(sigs_stft, kernel_size_scm_smoothing=3, eig_val_ratio_th=.9,):
    """
       Computes smoothed spatial covariance matrices (SCMs) from the input STFT signals and determines the dominant time-frequency mask.

       Args:
           sigs_stft (np.ndarray): Multichannel STFT signals with shape (channels, time, frequency).
           kernel_size_scm_smoothing (int): Size of the smoothing kernel for SCM calculation.
           eig_val_ratio_th (float): Threshold for the eigenvalue ratio to determine dominance.

       Returns:
           scms (np.ndarray): Smoothed SCMs with shape (frequencies, time, channels, channels).
           dominant (np.ndarray): Boolean mask indicating dominant time-frequency bins.
    """
    scms = np.einsum('c t f, d t f -> f t c d', sigs_stft, sigs_stft.conj())
    scms = [
        fftconvolve(
            np.pad(
                freq,
                (
                    (kernel_size_scm_smoothing // 2, kernel_size_scm_smoothing // 2),
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
                frame,
                (
                    (kernel_size_scm_smoothing // 2, kernel_size_scm_smoothing // 2),
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
        for frame in scms
    ]
    scms = rearrange(scms, 't f c d -> f t c d')
    scms = np.asarray(scms)
    dominant = compute_dominant_mask(scms, eig_val_ratio_th)
    return scms, dominant

def get_dominant_time_frequency_mask(sigs_stft, kernel_size_scm_smoothing=3, eig_val_ratio_th=0.9):
    """
      Computes a dominant time-frequency mask for multichannel STFT signals.

      For each time frame, this function calculates spatial covariance matrices (SCMs) over a local window,
      smooths them, and determines the dominance of the principal eigenvalue compared to the second largest.
      A time-frequency bin is marked as dominant if the ratio of the second to the largest eigenvalue is below a threshold,
      and the largest eigenvalue exceeds a minimum threshold.
      Args:
          sigs_stft (np.ndarray): Multichannel STFT signals with shape (channels, time, frequency).
      Returns:
          np.ndarray: Boolean mask of shape (time, frequency) indicating dominant time-frequency bins.
      """
    dominant = np.zeros_like(sigs_stft[0], bool)
    eig_val_mem = np.zeros_like(sigs_stft[0])
    sigs_stft_ = np.pad(sigs_stft, ((0, 0), (1, 1), (0, 0)), mode='edge')
    for i in range(1, sigs_stft_.shape[1] - 1):
        scms = np.einsum('ctf, dtf -> fcd', sigs_stft_[:, i - 1:i + 2], sigs_stft_[:, i - 1:i + 2].conj())
        scms = fftconvolve(
            np.pad(
                scms,
                (
                    (kernel_size_scm_smoothing // 2, kernel_size_scm_smoothing // 2),
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
        dominant[i - 1] = (dominance >= eig_val_ratio_th)
        eig_val_mem[i - 1] = eig_vals[..., -1]
    eig_val_th = 10 * np.min(eig_val_mem)
    dominant *= (eig_val_mem > eig_val_th)
    return dominant

def compute_steering_and_similarity_masks(sigs_stft, num_channels, tdoas_segment, k, fft_size, th=.3):
    """
    Computes Instantaneous SCM and reference SCM from the steering vector based on TDOA (Time Difference of Arrival).
    Args:
        sigs_stft (np.ndarray): STFT of the multichannel signals, shape (channels, time, frequency).
        sigs (np.ndarray): Multichannel time-domain signals.
        tdoas_segment (list): List of TDOA values for each segment.
        k (np.ndarray): Frequency bin indices.
        fft_size (int): FFT size used for STFT.
        th (float): Threshold for similarity mask.
    Returns:
        np.ndarray: Array of similarity masks for each segment, shape (num_segments, time, frequency).
        np.ndarray: Instantaneous SCM for the signals, shape (time, frequency, channels, channels).
    """
    all_masks = []
    inst_scm = np.einsum('ctf, dtf -> tfcd', sigs_stft, sigs_stft.conj())
    inst_scm /= abs(inst_scm) + 1e-18
    for t in tdoas_segment:
        t_ = np.pad(t[num_channels - 1], (1, 0))
        steer = np.exp(-1j * 2 * np.pi * k[:, None] / fft_size * t_)
        ref_scm = np.einsum('fc, fd -> fcd', steer, steer.conj())
        sim = correlation_matrix_distance(ref_scm, inst_scm)
        mask = sim < th
        all_masks.append(mask)
    masks = np.asarray(all_masks).astype(np.float64)
    return masks, inst_scm

def extract_segment_stft_and_context(seg_idx, segments, sigs, sigs_stft_complete, seg_tdoas, frame_shift,
                                     fft_size, context, max_diff_tmp_cl, max_offset=0):
    """
        Extracts the STFT and context information and applies wpe for a given segment.

        Args:
            seg_idx (int): Index of the segment to extract.
            segments (list): List of activity intervals for all segments.
            sigs (np.ndarray): Multichannel time-domain signals.
            sigs_stft_complete (np.ndarray): Complete STFT of the signals.
            seg_tdoas (list): List of TDOA values for each segment.
            max_offset (int): Current maximum offset across all segments.
            frame_shift (int): Frame shift used in STFT.
            fft_size (int): FFT size used in STFT.
            context (int): Number of samples to include as context before and after the segment.
            max_diff_tmp_cl (float): Maximum allowed TDOA difference for merging segments.

        Returns:
            sigs_stft (np.ndarray): STFT of the extracted segment with context.
            tdoas_segment (list): List of TDOA values for the segment and merged segments.
            activities (list): List of activity intervals for the segment and merged segments.
            onset (int): Onset frame index (STFT domain).
            offset (int): Offset frame index (STFT domain).
        """
    onset, offset = segments[seg_idx].intervals[0]
    if offset > max_offset:
        max_offset = offset
    onset = np.maximum(0, onset - int(context))
    offset = offset + int(context)
    act = pb.array.interval.zeros(sigs.shape[-1])
    act.add_intervals([slice(onset, offset), ])
    tdoas_segment = [seg_tdoas[seg_idx]]
    activities = [act, ]

    for other_seg in range(len(segments)):
        if seg_idx == other_seg:
            continue
        skip = False
        for t in tdoas_segment:
            if np.linalg.norm(
                    t - seg_tdoas[other_seg]) <= max_diff_tmp_cl:  # np.all(abs(t - seg_tdoas[other_seg]) < .5 ):
                skip = True
        if skip:
            continue
        other_act = np.asarray(segments[other_seg])
        if np.sum(act * other_act) > 0:
            tdoas_segment.append(seg_tdoas[other_seg])
            activities.append(other_act)
    onset = int(np.floor(onset // frame_shift))
    offset = int(np.ceil((offset - fft_size) / frame_shift))
    sigs_stft = sigs_stft_complete[:,
                onset:offset].copy()
    sigs_stft = wpe_v8(
        rearrange(sigs_stft, 'd t f -> f d t')
    )
    sigs_stft = rearrange(sigs_stft, 'f d t -> d t f')
    return sigs_stft, tdoas_segment, activities, onset, offset

def resolve_mask_ambiguities(masks, tdoas_segment, num_channels, k, fft_size, inst_scm, dominant):
    """
    Resolves ambiguities between masks by deciding, for each mask pair, which mask dominates for each time-frequency bin,
    based on the similarity of the reference SCMs to the current instantaneous SCMs. The SCM are set to one for
    the tf-bins in which both masks are active simultaneously when they are closer to inst_SCM than the other ref_SCM.
    Args:
        masks (np.ndarray): Array of masks (s, t, f).
        tdoas_segment (list): List of TDOA vectors for each segment.
        num_channels: Number of microphone channels in the recording
        k (np.ndarray): Frequency axis array.
        fft_size (int): FFT size
        inst_scm (np.ndarray): Instantaneous SCMs.
        dominant (np.ndarray): Dominance mask.
    Returns:
        np.ndarray: Adjusted masks after conflict resolution.
    """
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            t_i = np.pad(tdoas_segment[i][:num_channels - 1], (1, 0))
            t_j = np.pad(tdoas_segment[j][:num_channels - 1], (1, 0))
            steer_i = np.exp(-1j * 2 * np.pi * k[:, None] / fft_size * t_i)
            steer_j = np.exp(-1j * 2 * np.pi * k[:, None] / fft_size * t_j)
            ref_scm_i = np.einsum('fc, fd -> fcd', steer_i, steer_i.conj())
            ref_scm_j = np.einsum('fc, fd -> fcd', steer_j, steer_j.conj())
            decision_mask = correlation_matrix_distance(ref_scm_i, inst_scm) <= correlation_matrix_distance(ref_scm_j,
                                                                                                            inst_scm)
            both_mask = masks[i] * masks[j]
            both_mask = both_mask.astype(bool)
            masks[i, both_mask] = decision_mask[both_mask]
            masks[j, both_mask] = 1 - decision_mask[both_mask]
    masks *= dominant
    return masks


def cacgmm_mask_refinement(masks, sigs_stft, seg_acitivities, dominant, fft_size,  weight_constant_axis=-3,
                           max_val=0.8, num_iterations=10, track_noise_component=False):
    """
    Predicts time-frequency masks using the Complex Angular Central Gaussian Mixture Model (CACGMM).
    This function initializes and fits a CACGMM to the input STFT signals, using provided segment activities and a dominance mask.

    Args:
        masks (np.ndarray): Initial masks for each segment, shape (num_segments, time, frequency)
        sigs_stft (np.ndarray): Multichannel STFT signals with shape (channels, time, frequency).
        seg_acitivities (np.ndarray): Array indicating active frames for each segment.
        dominant (np.ndarray): Boolean mask indicating dominant time-frequency bins.
        fft_size (int): FFT size used for the STFT.

    Returns:
        np.ndarray: Predicted masks for each segment, shape (num_segments, time, frequency).
    """

    trainer = CACGMMTrainer()
    permutation_aligner = DHTVPermutationAlignment.from_stft_size(fft_size)
    input_mm = rearrange(sigs_stft, 'd t f -> f t d')
    masks *= seg_acitivities[..., None].astype(bool)
    init_masks = np.concatenate((masks, 1 - dominant[None]))
    init_masks = init_masks.astype(np.float64)
    init_masks[init_masks != 0] = max_val
    init_masks[init_masks == 0] = (1 - max_val) / (len(init_masks) - 1)
    init_masks = rearrange(init_masks, 'd t f -> f d t')

    cacgmm = trainer.fit(
        input_mm,
        initialization=init_masks,
        weight_constant_axis=weight_constant_axis,
        iterations=num_iterations,
        inline_permutation_aligner=permutation_aligner
    )
    refined_masks = cacgmm.predict(input_mm)
    refined_masks = rearrange(refined_masks, 'f s t -> s t f')
    if track_noise_component:
        mask_activities = np.sum(refined_masks, axis=(1,2))
        noise_idx = np.argmax(mask_activities)
        refined_masks = np.delete(refined_masks, noise_idx, axis=0)
    else:
        refined_masks = refined_masks[:-1,...]
    return refined_masks
