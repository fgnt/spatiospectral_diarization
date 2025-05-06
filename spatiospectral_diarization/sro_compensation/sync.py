import numpy as np
import paderbox as pb
from paderwasn.synchronization.sro_estimation import DynamicWACD
from paderwasn.synchronization.sync import compensate_sro
from paderwasn.synchronization.utils import VoiceActivityDetector


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
    for ch_id in range(1, len(sigs)):
        energy = np.sum(
            pb.array.segment_axis(
                sigs[ch_id][sigs[ch_id] > 0], 1024, 256, end='cut'
            ) ** 2, axis=-1
        )
        th = np.min(energy[energy > 0])
        vad = VoiceActivityDetector(10 * th, len_smooth_win=0)
        act = vad(sigs[ch_id])
        sro = sro_estimator(sigs[ch_id], sigs[0], act, ref_act)
        sros.append(sro)
    return sros


def compensate_for_sros(sigs, sros):
    ref_len = len(sigs[0])
    synced_sigs = np.zeros((len(sigs), ref_len))
    synced_sigs[0] = sigs[0].copy()
    for ch_id, sro in enumerate(sros):
        synced_sig = compensate_sro(sigs[ch_id + 1], sro)
        if len(synced_sig) > synced_sigs.shape[-1]:
            synced_sigs[ch_id + 1] = synced_sig[:synced_sigs.shape[-1]]
        elif len(synced_sig) < synced_sigs.shape[-1]:
            synced_sigs[ch_id + 1, :len(synced_sig)] = synced_sig
        else:
            synced_sigs[ch_id + 1] = synced_sig
    return synced_sigs
