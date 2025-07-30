import hashlib
from copy import deepcopy
from pathlib import Path
import shutil
from subprocess import run

import click
import numpy as np

import paderbox as pb
from paderbox.io.download import download_file
from paderbox.io.download import extract_file
from paderbox.io.download import download_file_list
from tqdm import tqdm


def check_md5(file_name, check_sum, blocksize=8192):
    hasher = hashlib.md5()
    with open(file_name, 'rb') as file:
        block = file.read(blocksize)
        while len(block) > 0:
            hasher.update(block)
            block = file.read(blocksize)
    assert hasher.hexdigest() == check_sum, \
        'md5sum not equal. Please check the download and retry.'


def download_files(files, database_path):
    progress = tqdm(
        files, desc="{0: <25s}".format('Download files')
    )
    downloaded_files = []
    for file in progress:
        file_name = file.split('/')[-1]
        downloaded_files.append(
            download_file(
                file,
                database_path / file_name
            )
        )
    return downloaded_files


def extract_files(downloaded_files):
    progress = tqdm(
        downloaded_files, desc="{0: <25s}".format('Extract files')
    )
    for file in progress:
        extract_file(file)


def download_aux_files(database_path):
    print('Download auxiliary material of LibriWASN')
    files = [
        'https://zenodo.org/record/10952434/files/ccby4.txt',
        'https://zenodo.org/record/10952434/files/LibirWASN200_Picture.png',
        'https://zenodo.org/record/10952434/files/LibriWASN200_Positions.pdf',
        'https://zenodo.org/record/10952434/files/LibriWASN200_Setup.png',
        'https://zenodo.org/record/10952434/files/Positions200.pdf',
        'https://zenodo.org/record/10952434/files/Positions800.pdf',
        'https://zenodo.org/record/10952434/files/readme.txt',
    ]
    download_file_list(files, database_path)
    target_dir = database_path / 'LibriWASN' / 'aux_files'
    target_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        file_name = file.split('/')[-1]
        shutil.move(database_path / file_name, target_dir / file_name)


def download_libriwasn200(database_path):
    print('Download LibriWASN 200')
    files = [
        'https://zenodo.org/record/10952434/files/LibriWASN_200_0L.zip',
        'https://zenodo.org/record/10952434/files/LibriWASN_200_0S.zip',
        'https://zenodo.org/record/10952434/files/LibriWASN_200_OV10.zip',
        'https://zenodo.org/record/10952434/files/LibriWASN_200_OV20.zip',
        'https://zenodo.org/record/10952434/files/LibriWASN_200_OV30.zip',
        'https://zenodo.org/record/10952434/files/LibriWASN_200_OV40.zip'
    ]
    md5_checksums = [
        '2327be91485110031181782c1605bd86',
        '531549b8528a10e1eb9ee6ad9f800073',
        'b6eecbd9dd4a1a2074b7cd681b722c5c',
        '1a8ba4ab2d74300fbe8fdb1de31d3379',
        '8cc0d8561ac9571561e8d5ed628404db',
        '9d33cdaea1b1c968d8f885c80ce4d761'
    ]
    downloaded_files = download_files(files, database_path)
    for file, check_sum in zip(downloaded_files, md5_checksums):
        check_md5(file, check_sum)
    print('Extract LibriWASN 200')
    extract_files(downloaded_files)
    shutil.move(database_path / 'LibriWASN' / '200',
                database_path / 'LibriWASN' / 'libriwasn_200')


def download_libriwasn800(database_path):
    print('Download LibriWASN 800')
    files = [
        'https://zenodo.org/record/10952434/files/LibriWASN_800_0L.zip',
        'https://zenodo.org/record/10952434/files/LibriWASN_800_0S.zip',
        'https://zenodo.org/record/10952434/files/LibriWASN_800_OV10.zip',
        'https://zenodo.org/record/10952434/files/LibriWASN_800_OV20.zip',
        'https://zenodo.org/record/10952434/files/LibriWASN_800_OV30.zip',
        'https://zenodo.org/record/10952434/files/LibriWASN_800_OV40.zip'
    ]
    md5_checksums = [
        'e9cbaf2c4e35aeea0ac14c7edf9c181f',
        'aa8442d009dd669c14f680ba20e2143f',
        '5e36a163669bbfaad01c617a6f7e4696',
        'f8efb703b0dca20a03bbcb2f9ef07a07',
        'c76c0a22da2e7299b06fe239b7681615',
        'd3fdc9b79c33025eb0fa353e31a80c71'
    ]
    downloaded_files = download_files(files, database_path)
    for file, check_sum in zip(downloaded_files, md5_checksums):
        check_md5(file, check_sum)
    print('Extract LibriWASN 800')
    extract_files(downloaded_files)
    shutil.move(database_path / 'LibriWASN' / '800',
                database_path / 'LibriWASN' / 'libriwasn_800')


def download_libricss(database_path):
    print('Download LibriCSS')
    link = 'https://docs.google.com/uc?export=download' \
           '&id=1Piioxd5G_85K9Bhcr8ebdhXx0CnaHy7l'
    if shutil.which('gdown'):
        run(['gdown', link, '-Olibricss.zip'])
    else:
        raise OSError(
            'gdown is not installed, You have to install it'
            ' to be able to download LibriCSS.'
        )
    print('Extract LibriCSS')
    extract_file(database_path / 'libricss.zip')
    shutil.move(database_path / 'for_release', database_path / 'LibriCSS')

def read_tsv(path):
    return [line.split('\t') for line in path.read_text().split('\n') if line]


def create_database(database_path: Path, sample_rate=16000):
    def get_info_libriwasn(audio_root):
        file_paths = {}
        num_samples = {}
        for file in audio_root.glob('*.wav'):
            file = str(file)
            file_id = file.split('/')[-1]
            if file_id == 'raw_recording.wav':
                continue
            device_id = file_id.split('_')[0]
            if device_id == 'Raspi':
                device_id = file_id.split('_')[2]
            num_samples[device_id] = pb.io.audioread.audio_length(file)
            file_paths[device_id] = file
        return file_paths, num_samples

    overlap_conditions = ['0L', '0S', 'OV10', 'OV20', 'OV30', 'OV40']
    database = {}
    libricss_set = database['libricss'] = {}
    libriwasn_200_set = database['libriwasn200'] = {}
    libriwasn_800_set = database['libriwasn800'] = {}
    for cond in overlap_conditions:
        for meeting_path in (database_path / 'LibriCSS' / cond).glob('*'):
            meeting_id = meeting_path.name
            session_id = meeting_id.split('_')[-2]
            onsets = []
            num_samples = []
            transcriptions = []
            speaker_ids = []
            source_ids = []
            meeting_info = read_tsv(
                meeting_path / 'transcription' / 'meeting_info.txt'
            )
            activity_ints = dict()
            for (start_time, end_time, speaker, utterance_id, transcription) \
                    in meeting_info[1:]:
                speaker_ids.append(speaker)
                if speaker not in activity_ints.keys():
                    activity_ints[speaker] = []
                onset = int(float(start_time) * sample_rate)
                length = int((float(end_time)-float(start_time))*sample_rate)
                source_ids.append(utterance_id)
                onsets.append(onset)
                num_samples.append(length)
                transcriptions.append(transcription)
                activity_ints[speaker].append(slice(onset, onset+length))

            activity = dict()
            for spk, intervals in activity_ints.items():
                act = pb.array.interval.zeros()
                act.add_intervals(intervals)
                activity[spk] = act.to_serializable()
            num_samples_obs = pb.io.audioread.audio_length(
                meeting_path / 'record' / 'raw_recording.wav'
            )
            num_samples_clean_obs = pb.io.audioread.audio_length(
                meeting_path / 'clean' / 'mix.wav'
            )
            num_samples_played = pb.io.audioread.audio_length(
                meeting_path / 'clean' / 'each_spk.wav'
            )
            libricss_set[f'{meeting_id}'] = {
                    'audio_path': {
                        'observation': str(
                            meeting_path / 'record' / 'raw_recording.wav'
                        ),
                        'clean_observation': str(
                            meeting_path / 'clean' / 'mix.wav'
                        ),
                        'played_signals': str(
                            meeting_path / 'clean' / 'each_spk.wav'
                        ),
                    },
                    'speaker_id': speaker_ids,
                    'source_id': source_ids,
                    'activity_orig': activity,
                    'onset': {'original_source': onsets},
                    'num_samples': {
                        'original_source': num_samples,
                        'observation': num_samples_obs,
                        'clean_observation': num_samples_clean_obs,
                        'played_signals': num_samples_played
                    },
                    'transcription': transcriptions,
                    'overlap_condition': cond,
                    'session': session_id
            }

            onsets_libriwasn = \
                [np.maximum(onset - 32000, 0) for onset in onsets]
            libriwasn_200_example = libriwasn_200_set[f'{meeting_id}'] = \
                deepcopy(libricss_set[f'{meeting_id}'])
            file_paths, num_samples = get_info_libriwasn(
                database_path / 'LibriWASN' / 'libriwasn_200'/
                cond / meeting_id / 'record'
            )
            libriwasn_200_example['audio_path']['observation'] = file_paths
            libriwasn_200_example['onset']['original_source'] = \
                onsets_libriwasn
            libriwasn_200_example['num_samples']['observation'] = num_samples

            libriwasn_800_example = libriwasn_800_set[f'{meeting_id}'] = \
                deepcopy(libricss_set[f'{meeting_id}'])
            file_paths, num_samples = get_info_libriwasn(
                database_path / 'LibriWASN' / 'libriwasn_800' /
                cond / meeting_id / 'record'
            )
            libriwasn_800_example['audio_path']['observation'] = file_paths
            libriwasn_800_example['onset']['original_source'] = \
                onsets_libriwasn
            libriwasn_800_example['num_samples']['observation'] = num_samples
    return database


@click.command()
@click.option(
    '--database_path',
    '-db',
    type=str,
    default='libriwasn/',
    help='Base directory for the databases. Defaults to "libriwasn/".'
)
@click.option(
    '--json_path',
    '-json',
    type=str,
    default='libriwasn.json',
    help='Path to the json-file to be created as database descriptor'
)
def main(database_path, json_path):
    database_path = Path(database_path).expanduser().absolute()
    download_aux_files(database_path)
    download_libriwasn200(database_path)
    download_libriwasn800(database_path)
    download_libricss(database_path)
    print('Downloaded and extracted all files. Creating JSON descriptor.')
    json_db = create_database(database_path)
    json_db = {
        'datasets': json_db,
    }
    pb.io.dump_json(json_db, json_path, create_path=True, indent=4)
    print(f'Wrote {json_path}')


if __name__ == '__main__':
    main()
