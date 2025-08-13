# spatiospectral_diarization

# Combining local, spatial segmentation and global, embedding-based speaker assignment for diarization
Diarization is the task of determining "who spoke when" in a given audio recording.
Current popular approaches make use of a hybrid approach using a 
local segmentation module followed by a global speaker assignment which assigns the 
respective speaker identity to each segment. 
This repository implements a spatio-spectral diarization pipeline that makes use of the same 
structure, while replacing the local segmentation stage with a TDOA-based spatial 
segmentation module, as introduced in [_Spatio-spectral diarization of meetings by combining TDOA-based segmentation and speaker embedding-based clustering_](https://arxiv.org/abs/2506.16228).
 
The segmentation module is based on the spatial diarization pipeline proposed in
"_Spatial Diarization for Meeting Transcription with Ad-Hoc Acoustic Sensor Networks_, Tobias Gburrek, 
Joerg Schmalenstroer, Reinhold Haeb-Umbach, 2023 Asilomar Conference" [[link]](https://arxiv.org/abs/2311.15597)


> **_NOTE:_**
This repository is still undergoing changes. While the diarization pipeline is in a functional state and can be applied 
to arbitrary multi-channel recordings, the documentation is still incomplete and the code is undergoing revision for sake
of clarity and usability.
Therefore, the repository will still be undergoing some changes over the next few weeks.

# Content
- A multi-channel, spatio-spectral diarization pipeline
  - A spatial multi-channel segmentation module utilizing time difference of arrival (TDOA) features
  - Beamforming and TDOA segment refinement to remove segments corresponding to reflections
  - A global speaker assignment module using d-vector-based speaker embeddings obtained from the beamformed speech segments
- Scripts to reproduce the results of the reference publication [[link to the paper]](https://arxiv.org/abs/2506.16228)
  - Diarization of the LibriWASN and LibriCSS datasets
  - Evaluation in a semi-static meeting scenario 
- Modular design to enable further research and exchanging individual 
components of the pipeline

# Installation
After cloning the repository, you can install the package using pip:
```bash
git clone https://github.com/fgnt/spatiospectral_diarization.git
pip install spatiospectral_diarization
```

<!-- In the future, a direct installation from PyPI will be available as well.-->

See the code snippet below on how to directly apply the pipeline to a recording, 
or check the example notebook for further details on how to use the diarization pipeline and exchange parts of it 
(still WIP: to come in the next update).

# Applying the pipeline to a recording
We provide the full diarization pipeline pre-packaged into a single python class.
To apply the diarization pipeline to a multi-channel recording, you can use the following code snippet:

```python
from spatiospectral_diarization.spatio_spectral_pipeline import SpatioSpectralDiarization
import paderbox as pb

pipeline = SpatioSpectralDiarization(
    sample_rate=16000,  # Sample rate of the audio data
)

audio_signal = pb.io.load_audio('path/to/your/multi_channel_audio.wav')

output = pipeline(audio_signal)
```
The pipeline expects synchronized signals, both in terms of sampling rate offset (SRO)
and sampling time offset (STO). If you want to apply the pipeline to data obtained in a distributed
setup, e.g., from multiple recording devices, we recommend applying the synchronization modules from
[paderwasn](https://github.com/fgnt/paderwasn)   to the audio data before applying the diarization pipeline.

The pipeline outputs a dictionary containing the following entries:
- _diarization_estimate_: a dictionary containing all speakers with onsets and offsets of each speaker detected in the recording
- _activity_segments_: a list with all segments estimated in the spatial segmentation component
- _tdoa_vectors_: a list containing the corresponding average time differences of arrival (TDOAs) for each segment
- _embeddings_: The speaker embeddings for each segment 

For more details, on how to appply and the modify the pipeline, please refer to the **example notebook** in this repository.

# Reproducing the LibriWASN & LibriCSS results
> **_NOTE:_** Still Unndergoing final code revision. Data preparation scripts are available, call instruction on how to 
reproduce the results from the paper will be added in the next update.



# Citation
To cite this package, please refer to the following publication:

```
@misc{cordgburrek2025spatiospectral_diarization,
      title={Spatio-spectral diarization of meetings by combining {TDOA}-based segmentation and speaker embedding-based clustering}, 
      author={Tobias Cord-Landwehr and Tobias Gburrek and Marc Deegen and Reinhold Haeb-Umbach},
      year={2025},
      eprint={2506.16228},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2506.16228}, 
}
```
