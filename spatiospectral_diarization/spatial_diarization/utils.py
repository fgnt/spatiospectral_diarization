import numpy as np
import paderbox as pb
from paderbox.transform.module_stft import stft_frame_index_to_sample_index
from paderwasn.synchronization.utils import VoiceActivityDetector
from tcrl.utils import Kernel1D


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
        vad = VoiceActivityDetector(1.5 * th, len_smooth_win=0)
        act = vad(sig)
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
