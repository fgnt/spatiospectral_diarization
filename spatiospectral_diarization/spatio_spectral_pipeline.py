from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from spatiospectral_diarization.spatial_diarization.diarize import spatial_diarization
from spatiospectral_diarization.spatial_diarization.utils import get_position_candidates
from spatiospectral_diarization.embedding_based_clustering import embeddings_hdbscan_clustering

from spatiospectral_diarization.extraction.mask_estimation import (get_dominant_time_frequency_mask,
                                                                   extract_segment_stft_and_context,
                                                                   compute_smoothed_scms,
                                                                   compute_steering_and_similarity_masks,
                                                                   resolve_mask_ambiguities,
                                                                   cacgmm_mask_refinement,
                                                                   )
from spatiospectral_diarization.spatial_diarization.utils import  (convert_to_frame_wise_activities,
                                                                   channel_wise_activities,
                                                                   )
from spatiospectral_diarization.extraction.beamformer import time_varying_mvdr
from spatiospectral_diarization.utils import (postprocess_and_get_activities, assign_estimated_activities,
                                              extract_embeddings)

from speaker_reassignment.tcl_pretrained import PretrainedModel
import paderbox as pb
from utils import merge_overlapping_segments


@dataclass
class SpatioSpectralDiarization:
    """
    A class to represent a spatio-spectral diarization pipeline.

    Attributes:
        model: The model used for diarization.
        num_classes: The number of classes (speakers) to be identified.
        iterations: The number of iterations for the training process.
        saliency: Optional saliency map for enhancing features.
    """
    embedding_extractor: callable = PretrainedModel()
    vad_module: callable = channel_wise_activities
    stft_params_gcc: ClassVar = {'stft_size': 4096, 'shift': 1024, 'fading': False, 'pad': False, 'window': 'hann'}
    stft_params_bf: ClassVar = {'stft_size': 1024, 'shift': 256, 'fading': False, 'pad': False, 'window': 'hann'}

    sample_rate: int = 16_000

    tdoa_settings: ClassVar = {'max_diff': 1.,
                               'search_range': 5,
                               'f_min': 125,
                               'f_max':None,
                               'avg_len_gcc': 4,
                               'distributed': False}

    segmentation_settings: ClassVar = {'max_diff_tmp_cl': 0.75, 'min_srp_peak_rato':.5,
                                       'max_temp_dist_cl':16}

    clustering: callable = embeddings_hdbscan_clustering
    only_spatial_dia: bool = False
    apply_cacgmm_refinement: bool = True


    def __post_init__(self):
        self.stft_gcc = pb.transform.module_stft.STFT(**self.stft_params_gcc)
        self.stft_bf = pb.transform.module_stft.STFT(**self.stft_params_bf)
        assert self.vad_module is not None or self.vad_indexer is not None, "Either vad_module or vad_indexer must be provided."
        return

    def normalize_recording(self, recording):
        return (recording - np.mean(recording, keepdims=True)) / np.linalg.norm(recording, keepdims=True)

    def apply_vad(self, recording):
        vad_indexer = self.vad_module(recording)
        return convert_to_frame_wise_activities(vad_indexer, th=0.5, frame_shift=self.stft_params_gcc['shift'],
                                                frame_size=self.stft_params_gcc['stft_size'])

    def spatial_segmentation(self, recording, vad_indexer):

        gcc_features = self.stft_gcc(recording)

        dominant_source_mask = get_dominant_time_frequency_mask(gcc_features)

        tdoa_candidates = get_position_candidates(
            gcc_features, vad_indexer, dominant_source_mask,
            **self.tdoa_settings
        )

        segments = temporally_constrained_clustering(tdoa_candidates, **self.segmentation_settings)

        segments = segment[::-1] # ouput of temporal clustering is in temporla reverse order

        segments, segment_tdoas = merge_overlapping_segments(segments, recording.shape[-1],
                                                             avg_len_gcc=self.tdoa_settings['avg_len_gcc'],
                                                             min_cl_segment=3,
                                                             distributed=self.tdoa_settings['distributed'],
                                                             max_diff_tmp_cl=self.tdoa_settings['max_diff_tmp_cl']
                                                             )
        return segments, segment_tdoas

    def segment_wise_beamforming_and_embedding_extraction(self, recording, segments, segment_tdoas):
        recording_stft = self.stft_bf(recording)
        fft_size = self.stft_params_bf['stft_size'],
        num_channels = recording.shape[0]
        embeddings = []
        seg_boundaries = []
        for cur_segment in range(len(segments)):
            segment_stft, tdoas_segment, activities, onset, offset = extract_segment_stft_and_context(cur_segment, segments,
                                                                                                   sigs,
                                                                                                   recording_stft,
                                                                                                   seg_tdoas,
                                                                                                   frame_shift=self.stft_params_bf['shift'],
                                                                                                   fft_size=fft_size,
                                                                                                   max_diff_tmp_cl=self.tdoa_settings['max_diff_tmp_cl'],
                                                                                                   context=3.self.sample_rate)
            scms, dominant = compute_smoothed_scms(segment_stft)

            k = np.arange(fft_size // 2 + 1)
            """ Compute masks, postprocess the masks and activities"""
            masks, inst_scm = compute_steering_and_similarity_masks(sigs_stft, sigs, tdoas_segment, k, fft_size )
            masks = resolve_mask_ambiguities(masks, tdoas_segment, num_channels, k, fft_size, inst_scm, dominant)
            masks, seg_activities, tdoas_reduced, phantom = postprocess_and_get_activities(masks, tdoas_segment)
            if phantom:
                continue  # skip segments of phantom speakers


            if self.apply_cacgmm_refinement:
                masks = cacgmm_mask_refinement(masks, sigs_stft, seg_activities, dominant, fft_size, logger)
                masks, seg_activities, _, phantom = postprocess_and_get_activities(masks, tdoas_segment)
                if phantom:
                    continue  # skip segments of phantom speakers


            else:
                ##################################################
                """ Compute masks, postprocess the masks and activities"""
                masks, inst_scm = compute_steering_and_similarity_masks(sigs_stft, sigs, tdoas_reduced, k, fft_size)
                masks = resolve_mask_ambiguities(masks, tdoas_reduced, sigs, k, fft_size, inst_scm, dominant)
                masks, seg_activities, tdoas_reduced, phantom = postprocess_and_get_activities(masks, tdoas_reduced, act=0.3)
                if phantom:
                    continue  # skip segments of phantom speakers

            """Beamform the signals using the masks"""
            sig_segs, seg_onsets = time_varying_mvdr(sigs_stft, rearrange(masks, 's t f -> s f t'),
                                                     seg_activities.astype(bool), wpe=False)
            # Extract the speaker embedding from each segment
            embeddings, seg_boundaries = extract_embeddings(embeddings, seg_boundaries, sig_segs, seg_onsets,
                                                            self.embedding_extractor, onset, frame_shift=self.stft_params_bf['shift'],)
        return embeddings, seg_boundaries

    def __call__(self, recording, vad_indexer=None):
        """
        Process an example through the diarization pipeline.

        Args:
            example: A dictionary containing the audio data and metadata.

        Returns:
            A dictionary with processed features and metadata.
        """

        recording = self.normalize_recording(recording)



        if vad_indexer is not None:
            vad_indexer = convert_to_frame_wise_activities(vad_indexer)
        elif self.vad_module is not None:
            vad_indexer = self.apply_vad(recording)
        else:
            vad_indexer = np.ones_like(recording)

        segments, segment_tdoas = self.spatial_segmentation(recording, vad_indexer)



        if self.only_spatial_dia:
            est_activities_spatial, labels, num_spk = spatial_diarization(distributed=self.tdoa_settings['distributed'], segment_tdoas, segments, recording,
                                                                          dilation_len_spatial=32001,
                                                                          dilation_len_spatial_add=8001)
            spatial_diarization = {spk: pb.array.interval.ArrayInterval(act.astype(bool))
                                     for spk, act in enumerate(est_activities_spatial)}
            return {'diarization_estimate': spatial_diarization,
                    'segments':segments,
                    'segment_tdoas': segment_tdoas
                    'num_spk': num_spk}

        #Obtain segment-wise masks apply beamforming from TDOA segments, and extract embeddings
        embeddings, seg_boundaries)= self.segment_wise_beamforming_and_embedding_extraction(recording, segments,
                                                                                                         segment_tdoas)

        # Perform Clustering on the extracted embeddings
        labels, activities, embeddings = self.clustering(embeddings, seg_boundaries)
        est_activities = assign_estimated_activities(labels, activities, embeddings, recording.shape[-1])

        diarization_estimate = {spk: pb.array.interval.ArrayInterval(act.astype(bool))
                                     for spk, act in enumerate(est_activities)}
        return {
            'diarization_estimate': diarization_estimate,
            'embeddings': embeddings
            'segments': segments,
            'segment_tdoas': segment_tdoas
            'num_spk': len(set(labels))
        }

