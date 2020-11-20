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
    ap.add_argument(
        "-n",
        "--normalize",
        default=False,
        type=bool,
        required=False,
        help="Normalize dataset (True/False)",
    )
    ap.add_argument(
        "-s",
        "--standardize",
        default=False,
        type=bool,
        required=False,
        help="Standardize dataset (True/False)",
    )
    ap.add_argument(
        "-b",
        "--best-base",
        type=str,
        choices=["best", "any"],
        default="any",
        required=False,
        help="Best base classifier parameters {best, any}",
    )
    args = ap.parse_args()
    return (
        args.host,
        args.model,
        args.set_of_files,
        args.normalize,
        args.standardize,
        args.best_base,
    )


(
    host,
    model,
    set_of_files,
    normalize,
    standardize,
    best_base,
) = parse_arguments()
datasets = Datasets(False, False, set_of_files)
clf = None
experiment = Experiment(
    random_state=1, model=model, host=host, set_of_files=set_of_files
)
experiment.set_base_params(best_base)
for dataset in datasets:
    print(f"-Grid search on {dataset[0]}")
    experiment.grid_search(dataset[0], normalize, standardize)
