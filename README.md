# spatiospectral_diarization

# Combining local, spatial segmentation and global, embedding-based speaker assignment for diarization
Diarization is the task of determining "who spoke when" in a given audio recording.
Current popular approaches make use of a hybrid approach using a 
local segmentation module followed by a global speaker assignment which assigns the 
respective speaker identity to each segment. 
This repository implements a spatio-spectral diarization pipeline that makes use of the same 
structure, while replacing the local segmentation stage with a TDOA-based spatial 
segmentation module.
 
The segmentation module is based ont he spatial diarization pipeline proposed in
""


> **_NOTE:_**
This repository is a work in progress. While all essential building blocks are already incorporated, 
the repository will still be undergoing significant and fundamental changes over the next few weeks.
In addition, code for database preparation and evaluation is still under development and missing in the current
stage.
Use at your own risk.

# Content
- A multi-channel, spatio-spectral diarization pipeline
  - A spatial multi-channel segmentation module utilizing time difference of arrival (TDOA) features
    - A global speaker assignment module using d-vector-based speaker embeddings
- Scripts to reproduce the results of the reference publication
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

In the future, a direct installation from PyPI will be available as well.

See below and check the example notebook for further details on hot to use the 
diarization pipeline and exchange parts of it.

# Applying the pipeline to a recording
We provide the full pipeline pre-packaged into a single Python class.
To apply the diarization pipleine to a multi-channel recording, you can use the following code snippet:

```python
from spatiospectral_diarization.spatio_spectral_pipeline import SpatioSpectralDiarization
import paderbox as pb

pipeline = SpatioSpectralDiarization(
    sample_rate=16000,  # Sample rate of the audio data
    num_channels=4,     # Number of channels in the audio data
    embedding_model='path/to/your/speaker_embedding_model',  # Path to the pre-trained speaker embedding model
)

audio_signal = pb.io.load_audio('path/to/your/multi_channel_audio.wav')

output = pipeline(audio_signal)
```
The pipeline expects synchronized signals both in terms of sampling rate offset (SRO)
and sampling time offset (STO). If you want to apply the pipeline to data obtained in a distributed
setup, e.g. from multiple recoding devices, we recommend applying the synchronization modules from
paderwasn [TODO link]  to the audio data before applying the diarization pipeline.


# Reproducing the LibriWASN & LibriCSS results
> **_NOTE:_** Currently, only the evaluation scripts are available, the database preparation scripts and call instructions
> are still under development.

## LibriCSS & LibriWASN

## Semi-static meeting scenario


# Citation
To cite the spatio-spectral diarization pipeline, please cite the following publication:

```
@InProceedings{cordgburrek2025spatiospectral_diarization,
  Title                    = {Spatio-spectral diarization of meetings by combining {TDOA}-based segmentation and speaker embedding-based clustering},
  Author                   = {Cord-Landwehr, Tobias and Gburrek, Tobias and Deegen, Marc and Haeb-Umbach, Reinhold},
  Booktitle                = {Proc. of Interspeech 2025},
  Year                     = {2025},
  Month                    = {Aug}
}
```