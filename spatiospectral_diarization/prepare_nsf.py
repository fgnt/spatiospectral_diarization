import argparse
import sys
import paderbox as pb
from lazy_dataset.database import JsonDatabase
from pathlib import Path


def prepare_nsf(example, rttm_path, dset, sample_rate=16000):
    activities = {}
    rttm_path = rttm_path + f"notsofar_{dset}.rttm"
    for spk in example['activity'].keys():
        activities[spk] = pb.array.interval.core.ArrayInterval.from_serializable(example['activity'][spk])
    mode = 'a' if Path(rttm_path).is_file() else 'w'
    Path(rttm_path).open(mode).write(pb.array.interval.rttm.to_rttm_str({example["example_id"]: activities},
                                                                        sample_rate=sample_rate))
    Path(rttm_path).open("a").write("\n")
    return pb.array.interval.from_rttm(rttm_path)

def main(json_path="/scratch/hpc-prf-nt2/tvn/data/notsofar1/notsofar.json", rttm_path="/scratch/hpc-prf-nt2/deegen/data/notsofar/"):
    db = JsonDatabase(json_path)
    dsets = db.dataset_names
    for dset in dsets:
        data = db.get_dataset(dset)
        for example in data:
            prepare_nsf(example, rttm_path, dset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare NSF data for diarization.")
    parser.add_argument("json_path", type=str, help="Path to the JSON file containing the data.")
    parser.add_argument("rttm_path", type=str, help="Path to the output RTTM file.")

    if len(sys.argv) > 1:
        args = parser.parse_args()
        main(args.json_path, args.rttm_path)
    else:
        main()