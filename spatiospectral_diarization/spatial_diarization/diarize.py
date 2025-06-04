from einops import rearrange

import numpy as np
from sklearn.cluster import AgglomerativeClustering

import paderbox as pb
from nara_wpe.wpe import wpe_v8

from spatiospectral_diarization.spatial_diarization.srp_phat import get_position_candidates
from spatiospectral_diarization.spatial_diarization.cluster import (
    temporally_constrained_clustering,
    single_linkage_clustering
)
from spatiospectral_diarization.spatial_diarization.utils import (
    clusters_to_diary,
    diary_to_activities,
    frame_to_sample_activity,
    postprocess_activities,
    channel_wise_activities,
    convert_to_frame_wise_activities, erode, dilate
)

def tdoa_diarization(
        sigs, max_dist_merge=2, max_temp_dist_cl=32, min_srp_peak_rato=.75,
        frame_size=4096, frame_shift=1024, dilation_len=41, erosion_len=21
):
    sigs_stft = pb.transform.stft(sigs)
    sigs_stft = wpe_v8(
        rearrange(sigs_stft, 'd t f -> f d t')
    )
    sigs_stft = rearrange(sigs_stft, 'f d t -> d t f')
    sigs = pb.transform.istft(sigs_stft, num_samples=sigs.shape[-1])
    voice_activity = channel_wise_activities(sigs)
    frame_wise_voice_activity = convert_to_frame_wise_activities(
        voice_activity, frame_size=frame_size, frame_shift=frame_shift
    )
    sigs_stft = pb.transform.stft(
        sigs, frame_size, frame_shift, pad=False, fading=False
    )
    candidates = get_position_candidates(sigs_stft, frame_wise_voice_activity)
    temp_diary = temporally_constrained_clustering(
        candidates, max_dist=max_dist_merge,
        max_temp_dist=max_temp_dist_cl, peak_ratio_th=min_srp_peak_rato
    )
    clusters, _ = \
        single_linkage_clustering(temp_diary, max_dist=max_dist_merge**2)
    diary = clusters_to_diary(clusters, temp_diary)
    est_frame_actvitities = diary_to_activities(
        diary, sigs_stft.shape[1], dilation_len=dilation_len,
        erosion_len=erosion_len
    )
    est_activities = frame_to_sample_activity(
        est_frame_actvitities, frame_shift=frame_shift, frame_size=frame_size
    )
    tdoas = [np.median(entry[0], 0) for entry in diary]

    est_activities, tdoas = \
        postprocess_activities(est_activities, tdoas)
    return est_activities, np.asarray(tdoas)


def spatial_diarization(distributed, seg_tdoas, segments, sigs, dilation_len_spatial,
                        dilation_len_spatial_add):
    """
    Performs spatial diarization by clustering segments based on their TDOA (Time Difference of Arrival) values.

    Args:
        distributed (bool): If True, uses parameters suitable for distributed microphone setups.
        seg_tdoas (list or np.ndarray): List of TDOA values for each segment.
        segments (list): List of activity intervals for each segment.
        sigs (np.ndarray): Multichannel audio signals.
        gt_activities (list or np.ndarray): Ground truth speaker activities.
        dilation_len_spatial (int): Dilation length for post-processing the estimated activities.
        dilation_len_spatial_add (int): Additional dilation length for further post-processing.

    Returns:
        est_activities_spatial (np.ndarray): Estimated speaker activities after spatial clustering and post-processing.
        labels (np.ndarray): Cluster labels assigned to each segment.
        num_spk (int): Estimated number of speakers.
    """
    if distributed:
        labels = AgglomerativeClustering(n_clusters=None, distance_threshold=5, linkage='single').fit_predict(seg_tdoas)
        min_samples = 3
        for i in range(np.max(labels) + 1):
            if np.sum(labels == i) < min_samples:
                labels[labels == i] = -1
    else:
        labels = AgglomerativeClustering(n_clusters=None, distance_threshold=.25, linkage='single').fit_predict(
            seg_tdoas)
        min_samples = 3
        for i in range(np.max(labels) + 1):
            if np.sum(labels == i) < min_samples:
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
        np.array(dilate(pb.array.interval.ArrayInterval(act), dilation_len_spatial))
        # Kernel1D(dilation_len_spatial, kernel=np.max)(act)
        for act in est_activities
    ]
    est_activities = [
        np.array(erode(pb.array.interval.ArrayInterval(act), dilation_len_spatial))
        # Kernel1D(erosion_len_spatial, kernel=np.min)(act)
        for act in est_activities
    ]
    est_activities = [
        np.array(dilate(pb.array.interval.ArrayInterval(act), dilation_len_spatial_add))
        # Kernel1D(dilation_len_spatial, kernel=np.max)(act)
        for act in est_activities
    ]

    est_activities_spatial = np.asarray(est_activities)
    return est_activities_spatial, labels, num_spk

