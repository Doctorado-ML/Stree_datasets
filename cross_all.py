import argparse
from typing import Tuple

from experimentation.Sets import Datasets
from experimentation import Experiment


def parse_arguments() -> Tuple[str, str, str, bool, bool]:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-H",
        "--host",
        type=str,
        choices=["develop", "galgo"],
        required=False,
        default="develop",
    )
    ap.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["stree", "adaBoost", "bagging", "odte"],
        required=False,
        default="stree",
    )
    ap.add_argument(
        "-S",
        "--set-of-files",
        type=str,
        choices=["aaai", "tanveer"],
        required=False,
        default="aaai",
    )
    args = ap.parse_args()
    return (
        args.host,
        args.model,
        args.set_of_files,
    )


(
    host,
    model,
    set_of_files,
) = parse_arguments()
datasets = Datasets(False, False, set_of_files)
clf = None
experiment = Experiment(
    random_state=1, model=model, host=host, set_of_files=set_of_files
)
for dataset in datasets:
    print(f"-Cross validation on {dataset[0]}")
    experiment.cross_validation(dataset[0])
