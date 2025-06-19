from paderwasn.synchronization.sro_estimation import DynamicWACD
import numpy as np
import paderbox as pb
from paderbox.transform.module_stft import stft_frame_index_to_sample_index
from paderwasn.synchronization.utils import VoiceActivityDetector
from spatiospectral_diarization.utils import Kernel1D
import itertools
from scipy.signal import find_peaks

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
        vad = VoiceActivityDetector(7 * th, len_smooth_win=0) # oder 0?
        act = vad(sig)
        act = np.array(dilate(pb.array.interval.ArrayInterval(act), 3201))
        act = np.array(erode(pb.array.interval.ArrayInterval(act), 3201))
        activities[ch_id] = act[:len(sig)]
    return activities

def convert_to_frame_wise_activities(
        activities, th=.5, frame_size=1024, frame_shift=256
):
    frame_wise_activities = np.sum(
        pb.array.segment_axis(
            activities, length=frame_size, shift=frame_shift, end='cut'
        ), -1
    ) > th * frame_size
    return frame_wise_activities


def get_ch_pairs(num_chs):
    ch_pairs = []
    for i in range(num_chs):
        for j in range(i + 1, num_chs):
            ch_pairs.append((i, j))
    return ch_pairs


def erode(activity, kernel_size):
    activity_eroded = pb.array.interval.zeros(shape=activity.shape)
    for (onset, offset) in activity.normalized_intervals:
        onset += (kernel_size - 1) // 2
        onset = np.maximum(onset, 0)
        offset -= (kernel_size - 1) // 2
        offset = np.minimum(offset, activity.shape)
        activity_eroded.add_intervals([slice(onset, offset)])
    return activity_eroded


def dilate(activity, kernel_size):
    activity_dilated = pb.array.interval.zeros(shape=activity.shape)
    for (onset, offset) in activity.normalized_intervals:
        onset -= (kernel_size - 1) // 2
        onset = np.maximum(onset, 0)
        offset += (kernel_size - 1) // 2
        offset = np.minimum(offset, activity.shape)
        activity_dilated.add_intervals([slice(onset, offset)])
    return activity_dilated


def clusters_to_diary(clusters, temp_diary):
    diary = []
    for cluster in clusters:
        entry = ([], [])
        for member in cluster:
            tdoas, frame_ids, _ = temp_diary[member]
            entry = (entry[0] + tdoas, entry[1] + frame_ids)
        diary.append(entry)
    diary = sorted(diary, key=lambda x: len(x[1]), reverse=True)
    return diary


def diary_to_activities(
        diary, num_frames, avg_len=4, dilation_len=41,
        erosion_len=21, min_act=32
):
    diary = [entry for entry in diary if len(entry[1]) >= min_act]
    est_frame_actvitities = np.zeros((len(diary), num_frames), bool)
    for spk_id, entry in enumerate(diary):
        for frame in entry[1]:
            est_frame_actvitities[spk_id, frame - avg_len + 1:frame + 1] = 1
    est_frame_actvitities = [
        Kernel1D(erosion_len, kernel=np.min)(Kernel1D(dilation_len, kernel=np.max)(a))
        for a in est_frame_actvitities
    ]
    return np.asarray(est_frame_actvitities)


def frame_to_sample_activity(frame_activity, frame_size=4096, frame_shift=1024):
    num_spk, num_frames = frame_activity.shape
    num_time_steps = num_frames * frame_shift + frame_size
    time_activity = np.zeros((num_spk, num_time_steps), bool)
    for spk_id, activity in enumerate(frame_activity):
        for onset, offset in pb.array.interval.ArrayInterval(activity).intervals:
            onset = stft_frame_index_to_sample_index(
                onset, window_length=frame_size, shift=frame_shift,
                pad=False, fading=False, mode='first'
            )
            offset = stft_frame_index_to_sample_index(
                offset, window_length=frame_size, shift=frame_shift,
                pad=False, fading=False, mode='last'
            )
            time_activity[spk_id][onset:offset] = 1
    return time_activity


def postprocess_activities(activities, tdoas):
    est_activities_ = []
    tdoas_ = []
    clusters = []
    for i, activity in enumerate(activities):
        for j, ref_activity in enumerate(activities[:i]):
            if np.sum(activity * ref_activity) / np.sum(activity) > .5 \
                    and np.sum(np.abs(tdoas[i] - tdoas[j]) <=2) >1:
                break
        else:
            est_activities_.append(activity)
            tdoas_.append(tdoas[i])
            clusters.append(i)
    return np.asarray(est_activities_), tdoas_

def get_gcpsd(fft_seg, fft_ref_seg):
    cpsd = np.conj(fft_ref_seg) * fft_seg
    phat = np.abs(fft_seg) * np.abs(fft_ref_seg)
    gcpsd = cpsd / np.maximum(phat, 1e-9)
    return gcpsd

def get_position_candidates(sigs_stft, frame_wise_activities, dominant, f_min=125, f_max=3500, search_range=200,
                            avg_len=4, num_peaks=5, sample_rate=16000, max_diff=2, upsampling=10,
                            max_concurrent=3, distributed=False):
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
    lags = np.arange(-search_range, search_range + 1 / upsampling, 1 / upsampling)
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
            gcpsd_buffer[k, -1] = gcpsd * dominant[l] # Multiply dominantn to filter out noise frequencies
            avg_gcpsd = np.mean(gcpsd_buffer[k], 0)
            avg_gcpsd[avg_gcpsd > 0.5 / avg_len] /= np.abs(avg_gcpsd[avg_gcpsd > 0.5 / avg_len])
            if k_min is not None:
                avg_gcpsd[:k_min] = 0.
            if k_max is not None:
                avg_gcpsd[k_max:] = 0.
            avg_gcpsd = np.concatenate(
                [avg_gcpsd[:-1],
                 np.zeros((upsampling - 1) * (len(avg_gcpsd) - 1) * 2),
                 np.conj(avg_gcpsd)[::-1][:-1]]
            )
            gcc = np.fft.ifftshift(np.fft.ifft(avg_gcpsd).real)
            search_area = \
                gcc[len(gcc)//2-search_range*upsampling:len(gcc)//2+search_range*upsampling+1]
            th = np.maximum(0.75 * np.max(search_area), 0)#2 * np.sqrt(np.mean(search_area[search_area > 0] ** 2))

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
                if distributed:
                    srps.append((taus,srp))
                elif np.any(np.abs(taus) >= .5):
                    srps.append((taus,srp))
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
        candidates.append((l, spk_pos))
    return candidates

def estimate_sros(sigs):
    sro_estimator = DynamicWACD()
    sros = []
    energy = np.sum(
        pb.array.segment_axis(sigs[0][sigs[0] > 0], 1024, 256, end='cut') ** 2,
        axis=-1
    )
    th = np.min(energy[energy > 0])
    vad = VoiceActivityDetector(3 * th, len_smooth_win=0)
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

