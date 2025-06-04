import numpy as np
import paderbox as pb


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
    # Raises an exception if np.inf and/or np.NaN was in target_psd_matrix
    # or noise_psd_matrix
    assert np.all(np.isfinite(sdrs)), sdrs
    return sdrs.real


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

