from copy import deepcopy
from spatiospectral_diarization.sro_compensation.sync import compensate_for_sros
import dataclasses
import numpy as np
import paderbox as pb
import logging
from pathlib import Path
import scipy


@dataclasses.dataclass
class ABCKernel1D:
    kernel_size: int
    # stride: int
    # dilation: int
    padding_mode: ['edge'] = 'edge'
    pad_position: ['pre', 'post', None] = 'pre'

    def __post_init__(self):
        assert self.kernel_size % 2 == 1, (self.kernel_size, 'must be odd.')
        assert self.pad_position in ['pre', 'post', None], self.pad_position

    def kernel_fn(self, x):
        """
        Do an operation that removes the last axis of x.
        e.g. `np.mean(x, axis=-1)`
        """
        raise NotImplementedError()

    def __call__(self, x):
        if self.pad_position == 'pre':
            shift = self.kernel_size // 2
            padding = [(0, 0)] * (x.ndim - 1) + [(shift, shift)]
            x = np.pad(x, padding, 'edge')

        y = self.kernel_fn(
            pb.array.segment_axis(x, self.kernel_size, 1, end='pad')
        )

        if self.pad_position == 'post':
            shift = self.kernel_size // 2
            padding = [(0, 0)] * (y.ndim - 1) + [(shift, shift)]
            y = np.pad(y, padding, 'edge')
        return y


@dataclasses.dataclass
class Kernel1D(ABCKernel1D):
    kernel: callable = np.mean

    def kernel_fn(self, x):
        return self.kernel(x, axis=-1)


@dataclasses.dataclass
class MaxThresholdKernel1D(ABCKernel1D):
    threshold: float = 0.2

    def kernel_fn(self, x):
        """
        >>> w2a = MaxThresholdKernel1D(3, threshold=5)
        >>> x = np.array([0, 0, 1, 2, 3, 4, 5, 6, 7, 3, 2, 1, 0, 4, 5])
        >>> x.shape
        (15,)
        >>> y = w2a(x)
        >>> y.shape
        (15,)
        >>> print(x, y.astype(int), sep='\\n')
        [0 0 1 2 3 4 5 6 7 3 2 1 0 4 5]
        [0 0 0 0 0 1 1 1 1 1 0 0 0 1 1]
        """
        x_max = np.max(x, axis=-1)
        return np.where(x_max >= self.threshold, True, False)


def reduction_max_threshold(x, axis=-1, threshold=0.2):
    assert axis == -1, axis
    x_max = np.max(x, axis=axis)
    return np.where(x_max < threshold, False, True)


def smooth(noisy, window=25, reduction=reduction_max_threshold, pre_pad=True):
    assert (window % 2) == 1, (window, 'must be odd.')

    if pre_pad:
        shift = window // 2
        padding = [(0, 0)] * (noisy.ndim - 1) + [(shift, shift)]
        noisy = np.pad(noisy, padding, 'edge')

    smoothed = reduction(
        pb.array.segment_axis(noisy, window, 1, end='pad'),
        axis=-1)

    if not pre_pad:
        shift = window // 2
        padding = [(0, 0)] * (noisy.ndim - 1) + [(shift, shift)]
        smoothed = np.pad(smoothed, padding, 'edge')
    return smoothed

def setup_logger(log_dir=None, log_level=logging.INFO):
    logger = logging.getLogger('spatiospectral_logger')
    logger.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler (optional, für Slurm-Logs)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / 'pipeline.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger





def solve_permutation(activities, ref_activities):
    if len(ref_activities) < len(activities):
        ref_activities = np.pad(
            ref_activities,
            ((0, len(activities) - len(ref_activities)), (0, 0)),
            'constant'
        )
    elif len(ref_activities) > len(activities):
        activities = np.pad(
            activities,
            ((0, len(ref_activities) - len(activities)), (0, 0)),
            'constant'
        )
    assert len(ref_activities) == len(activities), (
    len(ref_activities), len(activities))

    overlaps = np.zeros((len(activities), len(activities)))
    for i, act in enumerate(activities):
        for j, ref_act in enumerate(ref_activities):
            overlaps[i, j] = np.sum(act == ref_act)
    '''costs = []
    for permutation in permutations(np.arange(len(ref_activities))):
        cost = 0
        for j, i in enumerate(permutation):
            cost += overlaps[i, j]
            j += 1
        costs.append(cost)
    costs = np.asarray(costs)'''
    _, best_permutation = scipy.optimize.linear_sum_assignment(overlaps.T, maximize=True)
    '''all_permuations = \
        [permutation
         for permutation in permutations(np.arange(len(activities)))]
    best_permutation = all_permuations[np.argmax(costs)]'''
    return np.asarray(best_permutation)


def cos_dist(embed_1, embed_2):
    return (1 - embed_1 @ embed_2 / np.sqrt(embed_1 @ embed_1) / np.sqrt(embed_2 @ embed_2) ) / 2

def select_channels(channel_set: str) -> np.ndarray:
    """
    Returns an array with channel indices according to the given set.
    >>> select_channels("set1")
    array([1, 3, 4, 6])
    """
    sets = {
        "set1": np.array([1, 3, 4, 6]),
        "set2": np.array([0, 2, 5, 7]),
        "set3": np.array([0, 1, 2, 3]),
        "all": np.array([0, 1, 2, 3, 4, 5, 6, 7])
    }
    if channel_set not in sets:
        import warnings
        warnings.warn(f"Unknown channel-set'{channel_set}', using channels '[1, 3, 4, 6]'.")
        return sets["set1"]
    return sets[channel_set]

def load_signals(session: dict, channels: np.ndarray, setup: str, subset: str, logger) -> np.ndarray:
    """
    Loads the audio signals for a given session from the dataset "subset" and returns them as an array.
    """
    try:
        if setup == 'compact':
            if session['dataset'] == 'libricss':
                sigs = pb.io.load_audio(session['audio_path']['observation'])[channels]
            elif 'libriwasn' in subset:
                sigs = pb.io.load_audio(session['audio_path']['observation']['asnupb7'])
            elif 'train_set_240130.1_train' in subset:
                logger.info("Load NotSoFar signals")
                sigs = []
                for i in channels:
                    sigs.append(pb.io.load_audio(session['audio_path']['observation']['mc']['plaza_0'][i]))
                sigs = np.array(sigs)
            else:
                logger.error(f'Unknown dataset: {subset}')
                raise KeyError(f'Undefined dataset {subset}')
        elif setup == 'distributed':
            sig0 = pb.io.load_audio(session['audio_path']['observation']['Pixel6a'])
            sig1 = pb.io.load_audio(session['audio_path']['observation']['Pixel6b'])
            sig2 = pb.io.load_audio(session['audio_path']['observation']['Pixel7'])
            sig3 = -pb.io.load_audio(session['audio_path']['observation']['Xiaomi'])

            min_len_ = np.min([len(sig0), len(sig1), len(sig2), len(sig3)])
            sigs = np.vstack((sig0[:min_len_], sig1[:min_len_], sig2[:min_len_], sig3[:min_len_]))
            sros = estimate_sros(sigs)
            sigs = compensate_for_sros(sigs, sros)
        else:
            logger.error(f'Undefined Setup {setup}')
            raise KeyError(f'Undefined Setup {setup}')
    except Exception as e:
        logger.exception(f'Error loading audio data: {e}')
        raise
    return sigs

def merge_and_extract_segments(temp_diary, sigs, avg_len_gcc, min_cl_segment, distributed, max_diff_tmp_cl):
    """
    Merges overlapping segments from the same direction. For each segment, the corresponding activity interval and the
    median TDOA (Time Difference of Arrival).
    Args:
        temp_diary (list): List of segment entries, each containing TDOA values and frame indices.
        sig_len (int): Length of the signal in samples.
        avg_len_gcc (int): Average length of GCC (Generalized Cross-Correlation)-Buffer calculation.
        min_cl_segment (int): Minimum number of frames required for a segment to be considered.
        distributed (bool): Whether the setup is distributed (affects segment filtering).
        max_diff_tmp_cl (float): Maximum allowed difference between median TDOAs for merging.
    Returns:
        tuple: (segments, seg_tdoas)
            segments (list): List of activity intervals for each merged segment.
            seg_tdoas (list): List of median TDOA values for each merged segment.
    """
    temp_diary_ = deepcopy(temp_diary)
    seg_tdoas = []
    segments = []
    for i, entry in enumerate(temp_diary_):
        if not distributed:
            if np.all(abs(np.median(entry[0], 0)) < .2):
                continue # skip noise position "above" the microphone
        if len(entry[1]) <= min_cl_segment:
            continue
        med_tdoa = np.median(entry[0], 0)
        act = pb.array.interval.zeros(sigs.shape[-1])
        onset = np.maximum((np.min(entry[1]) - avg_len_gcc) * 1024, 0)
        offset = np.max(entry[1]) * 1024 + 4096
        act.add_intervals([slice(onset, offset), ])
        to_remove = []
        for o, other in enumerate(temp_diary_[i + 1:]):
            if np.linalg.norm(np.median(other[0], 0) - med_tdoa) <= max_diff_tmp_cl:
                other_act = pb.array.interval.zeros(sigs.shape[-1])
                onset = np.maximum((np.min(other[1]) - avg_len_gcc) * 1024, 0)
                offset = np.max(other[1]) * 1024 + 4096
                other_act.add_intervals([slice(onset, offset), ])
                if np.sum(np.array(act) * np.array(other_act)) > 0:
                    for t in other[0]:
                        entry[0].append(t)
                    for t in other[1]:
                        entry[1].append(t)
                    to_remove.append(i + 1 + o)
        for remove_id in to_remove[::-1]:
            temp_diary_.pop(remove_id)
        med_tdoa = np.median(entry[0], 0)
        act = pb.array.interval.zeros(sigs.shape[-1])
        onset = np.maximum((np.min(entry[1]) - avg_len_gcc) * 1024, 0)
        offset = np.max(entry[1]) * 1024 + 4096
        act.add_intervals([slice(onset, offset), ])
        segments.append(act)
        seg_tdoas.append(med_tdoa)
    return segments, seg_tdoas

def postprocess_and_get_activities(masks, tdoas_segment, k_min, k_max, act_th, min_len, dilation_len_beam,
                                   erosion_len_beam, additional_dilate, cacgmm_param, reduce_tdoas):
    """
    Post-processes time-frequency masks and extracts segment activities.
    This function takes a set of masks and their corresponding TDOA segments, applies morphological operations
    (dilation and erosion) to smooth the activity detection, and filters out segments that are too short or
    considered phantom.

    Args:
        masks (np.ndarray): Array of time-frequency masks for each segment.
        tdoas_segment (list): List of TDOA vectors for each segment.
        k_min (int): Minimum frequency bin index for activity calculation.
        k_max (int): Maximum frequency bin index for activity calculation.
        act_th (float): Activity threshold for segment detection.
        min_len (int): Minimum length for a segment to be considered valid.
        dilation_len_beam (int): Length of the dilation kernel for activity smoothing.
        erosion_len_beam (int): Length of the erosion kernel for activity smoothing.

    Returns:
        masks (np.ndarray): Reduced and post-processed masks.
        seg_acitivities (np.ndarray): Array of activities for each valid segment.
        tdoas_reduced (list): List of TDOA vectors for valid segments.
        phantom (bool): True if the current segment is considered phantom and should be skipped.
    """
    seg_acitivities = []
    masks_reduced = []
    tdoas_reduced = []
    phantom = False
    for s, mask in enumerate(masks):
        act = np.mean(mask[:, k_min:k_max], -1) > act_th
        act = Kernel1D(erosion_len_beam, kernel=np.min)(
            Kernel1D(dilation_len_beam, kernel=np.max)(act)
        )
        if np.sum(act) >= min_len:
            # todo: cacgmm param umbennen und additional dilate umbenennen?
            if additional_dilate:
                act = Kernel1D(31, kernel=np.max)(act)
                if cacgmm_param:
                    mask *= act[:, None]
            masks_reduced.append(mask)
            seg_acitivities.append(act)
            if reduce_tdoas:
                tdoas_reduced.append(tdoas_segment[s])
        else:
            if s == 0:
                phantom = True
    masks = np.asarray(masks_reduced)
    seg_acitivities = np.asarray(seg_acitivities)
    return masks, seg_acitivities, tdoas_reduced, phantom

def assign_estimated_activities(labels, activities_red, embeddings_red, sigs):
    """
    Takes labeled embeddings and their activities, and creates an array with the estimated activities of each speaker.

    For each speaker cluster, determines the activity. For segments labeled as outliers \(`-1`\), assigns them to
    the closest speaker cluster based on cosine distance of embeddings.

    Args:
        labels (np.ndarray): Cluster labels for each segment.
        activities_red (list of tuple): List of \(`onset`, `offset`\) tuples for each segment.
        embeddings_red (np.ndarray): Embeddings for each segment.
        num_spk (int): Number of detected speakers.
        sigs (np.ndarray): Signal array to determine the total number of frames.

    Returns:
        np.ndarray: Estimated activities matrix of shape (`num_spk`, `num_frames`).
    """
    num_spk = np.max(labels) + 1
    est_activities = np.zeros((num_spk, sigs.shape[-1]))
    for spk_id in range(num_spk):
        for i in range(len(activities_red)):
            if labels[i] == spk_id:
                on, off = activities_red[i]
                est_activities[spk_id, on:off] = 1
    for i in range(len(activities_red)):
        if labels[i] == -1:
            on, off = activities_red[i]
            dists = [cos_dist(embeddings_red[i], spk_emdeb) if labels[s] != -1 else 10
                     for s, spk_emdeb in enumerate(embeddings_red)]
            closest_spk = labels[np.argmin(dists)]
            est_activities[closest_spk, on:off] = 1
    return est_activities

def dump_rttm(est_diarization, path):
    """
    Saves the given diarization in RTTM format to the specified path.
    Args:
        est_diarization: Diarization result in the appropriate format.
        path: Target path for the RTTM file.
    """
    pb.array.interval.rttm.to_rttm(est_diarization, path)
    return

def extract_embeddings(embeddings, seg_boundaries, sig_segs, seg_onsets, embed_extractor, onset, frame_shift):
    """
       Extracts embeddings for each signal segment and appends them with their boundaries.
       Args:
           embeddings (list): List to store embeddings.
           seg_boundaries (list): List to store segment boundaries as (onset, offset) tuples.
           sig_segs (list): List of signal segments.
           seg_onsets (list): List of onset positions for each segment.
           embed_extractor (callable): Function to extract embedding from a signal.
           onset (int): Onset frame index.
           frame_shift (int): Frame shift used in STFT.
       Returns:
           tuple: Updated embeddings and segment boundaries.
       """
    for s, sig in enumerate(sig_segs):
        embedding = embed_extractor(sig)
        embeddings.append(embedding)
        seg_boundaries.append((onset * frame_shift + seg_onsets[s], onset * frame_shift + seg_onsets[s] + len(sig)))
    return embeddings, seg_boundaries
