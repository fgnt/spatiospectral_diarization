from dataclasses import dataclass
import padertorch as pt
import paderbox as pb

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
    embedding_extractor: ResNetEmbeddingExtractor
    vad_module: callable =None
    vad_indexer: np.ndarray = None
    stft_params_gcc: dict= {'stft_size': 4096, 'shift': 1024, 'fading': False, 'pad': False}
    stft_params_bf: dict = {'stft_size': 1024, 'shift': 256, 'fading': False, 'pad': False}


    only_spatial_model: bool = False
    apply_cacgmm_refinement: bool = True


    def __post_init__(self):
        self.stft_gcc = pb.transform.module_stft.STFT(**self.stft_params_gcc)
        self.stft_bf = pb.transform.module_stft.STFT(**self.stft_params_bf)
        assert self.vad_module is not None or self.vad_indexer is not None, "Either vad_module or vad_indexer must be provided."
        return

    def apply_vad(self, recording):

        vad_indexer = self.vad_module(recording)

        self.vad_indexer = to_frame_resolution(vad_indexer)

    def spatial_segmentation(self, recording):

        gcc_features = self.stft_gcc(recording)

        return segments

    def __call__(self, recording):
        """
        Process an example through the diarization pipeline.

        Args:
            example: A dictionary containing the audio data and metadata.

        Returns:
            A dictionary with processed features and metadata.
        """

        recording = self.normalize_recording(recording)

        if self.vad_module is not None:
            self.apply_vad(recording)

        segments = self.spatial_segmentation(recording)

        segments = self.refine_segments(segments, vad_indexer)

        mask_estimates = self.get_masks_from_segments()

        audio_estimates = self.apply_bf(recording, mask_estimates)

        embeddings = self.extract_embeddings(audio_estimates)



        return {
            'features': features,
            'speaker_id': example['speaker_id'],
            'example_id': example['example_id'],
            'num_speaker': len(example['speaker_id'])
        }

