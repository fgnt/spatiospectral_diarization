from einops import rearrange

import numpy as np
import paderbox as pb
from nara_wpe.wpe import wpe_v8

from spatial_diarization.diarization.srp_phat import get_position_candidates
from spatial_diarization.diarization.cluster import temporally_constrained_clustering
from spatial_diarization.diarization.utils import (
    clusters_to_diary,
    diary_to_activities,
    frame_to_sample_activity,
    postprocess_activities
)
from spatial_diarization.diarization.utils import (
    channel_wise_activities,
    convert_to_frame_wise_activities
)
from spatial_diarization.diarization.cluster import single_linkage_clustering


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
