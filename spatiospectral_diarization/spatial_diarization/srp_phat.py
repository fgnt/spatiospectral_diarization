import itertools

import numpy as np
from scipy.signal import find_peaks

from spatial_diarization.diarization.utils import get_ch_pairs


def get_gcpsd(fft_seg, fft_ref_seg):
    cpsd = np.conj(fft_ref_seg) * fft_seg
    phat = np.abs(fft_seg) * np.abs(fft_ref_seg)
    gcpsd = cpsd / np.maximum(phat, 1e-9)
    return gcpsd


def get_position_candidates(
        sigs_stft, frame_wise_activities, f_min=125, f_max=3500,
        search_range=200, avg_len=4, num_peaks=5, sample_rate=16000,
        max_diff=2, max_diff_same_pos=5, max_concurrent=3
):
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
    lags = np.arange(-search_range, search_range+1)

    candidates = []
    gcpsd_buffer = \
        np.zeros((len(ch_pairs), avg_len, frame_size // 2 + 1), np.complex128)
    for l in range(frame_wise_activities.shape[-1]):
        for k, (i, j) in enumerate(ch_pairs):
            gcpsd_buffer[k] = np.roll(gcpsd_buffer[k], -1, axis=0)
            gcpsd_buffer[k, -1] = 0
        if np.sum(frame_wise_activities[:, l]) == 0:
            continue
        gccs = []
        peak_tdoas = []
        peaks = []
        for k, (ref_ch, ch) in enumerate(ch_pairs):
            fft_seg = sigs_stft[ch, l]
            fft_ref_seg = sigs_stft[ref_ch, l]
            gcpsd = get_gcpsd(fft_seg, fft_ref_seg)
            gcpsd_buffer[k, -1] = gcpsd
            avg_gcpsd = np.mean(gcpsd_buffer[k], 0)
            if k_min is not None:
                avg_gcpsd[:k_min] = 0.
            if k_max is not None:
                avg_gcpsd[k_max:] = 0.
            avg_gcpsd = np.concatenate(
                [avg_gcpsd[:-1],
                 np.conj(avg_gcpsd)[::-1][:-1]],
                -1
            )
            gcc = np.fft.ifftshift(np.fft.ifft(avg_gcpsd).real)
            search_area = \
                gcc[fft_size//2-search_range:fft_size//2+search_range]

            th = 2 * np.sqrt(np.mean(search_area[search_area > 0] ** 2))
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
        for combination in itertools.product(*[np.arange(len(p)) for p in peaks[:num_chs-1]]):
            taus = []
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
            srps.append((taus, srp))
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
                if np.sum(abs(np.asarray(t) - np.asarray(taus)) == 0) <= 1:
                    to_keep.append(srp)
            srps = to_keep
        #spk_pos = [(pos[0][:num_chs-1], pos[1])for pos in spk_pos]
        candidates.append((l, spk_pos))
    return candidates
