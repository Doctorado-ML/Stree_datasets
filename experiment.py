import argparse
from typing import Tuple

from experimentation import Experiment
from experimentation.Database import Hyperparameters, Outcomes
from experimentation.Sets import Datasets


def parse_arguments() -> Tuple[str, str, str, str, str, bool, bool, dict]:
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
        "-e",
        "--experiment",
        type=str,
        choices=[
            "gridsearch",
            "gridbest",
            "crossval",
            "report_grid",
            "report_cross",
        ],
        required=True,
        help="Experiment: {gridsearch, gridbest, crossval, report_grid, "
        "report_cross}",
    )
    ap.add_argument(
        "-k",
        "--kernel",
        type=str,
        choices=[
            "linear",
            "poly",
            "rbf",
            "any",
        ],
        required=False,
        default="any",
        help="Kernel: {linear, poly, rbf, any} only used in gridsearch",
    )
    ap.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=False,
        help="Dataset name",
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
        "-x",
        "--excludeparams",
        default=False,
        required=False,
        action="store_true",
        help="Exclude parameters in reports",
    )
    ap.add_argument(
        "-t",
        "--threads",
        default=-1,
        type=int,
        required=False,
        help="Number of threads to use or -1 for available cores",
    )
    args = ap.parse_args()
    return (
        args.host,
        args.model,
        args.set_of_files,
        args.experiment,
        args.dataset,
        args.normalize,
        args.standardize,
        args.excludeparams,
        args.kernel,
        args.threads,
    )


(
    host,
    model,
    set_of_files,
    experiment_type,
    dataset,
    normalize,
    standardize,
    exclude_params,
    kernel,
    threads,
) = parse_arguments()

experiment = Experiment(
    random_state=1,
    model=model,
    host=host,
    set_of_files=set_of_files,
    kernel=kernel,
    threads=threads,
)
if experiment_type[0:6] == "report":
    bd = (
        Outcomes(host, model)
        if experiment_type == "report_cross"
        else Hyperparameters(host, model)
    )
    bd.report("all", exclude_params)
elif experiment_type == "gridsearch" or experiment_type == "gridbest":
    if experiment_type == "gridbest":
        experiment.set_base_params("best")
    if dataset == "all":
        # Only want it for the dataset names
        dt = Datasets(False, False, set_of_files)
        for dataset in dt:
            print(f"Processing dataset: {dataset[0]}...")
            experiment.grid_search(dataset[0], normalize, standardize)
    else:
        experiment.grid_search(dataset, normalize, standardize)
elif experiment_type == "crossval":
    if dataset == "all":
        # Only want it for the dataset names
        dt = Datasets(False, False, set_of_files)
        for dataset in dt:
            print(f"Processing dataset: {dataset[0]}...")
            experiment.cross_validation(dataset[0])
    else:
        experiment.cross_validation(dataset)
