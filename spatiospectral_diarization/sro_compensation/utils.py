from scipy.signal import remez, freqz, lfilter


def correct_polarity(sigs, polarities):
    for i in range(len(sigs)):
        sigs[i] *= polarities[i]
    return sigs


def apply_high_pass(sigs):
    taps = remez(513, [0, 70, 125, 8000], [0, 1], weight=[1, 10], fs=16000)
    sigs_hp = [lfilter(taps, [1], sig) for sig in sigs]
    sigs = sigs_hp
    return sigs
