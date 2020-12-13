import argparse
from typing import Tuple
from experimentation.Sets import Datasets
from experimentation.Utils import TextColor
from experimentation.Database import MySQL

models = ["stree", "adaBoost", "bagging", "odte"]


def parse_arguments() -> Tuple[str, str, str, bool, bool]:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-m",
        "--model",
        type=str,
        choices=["any"] + models,
        required=False,
        default="any",
    )
    ap.add_argument(
        "-x",
        "--excludeparams",
        default=False,
        required=False,
        action="store_true",
        help="Exclude parameters in reports",
    )
    args = ap.parse_args()
    return (
        args.model,
        args.excludeparams,
    )


def report_header_content(title):
    length = sum(lengths) + len(lengths) - 1
    output = "\n" + "*" * length + "\n"
    title = title + f" -- {classifier} classifier --"
    num = (length - len(title) - 2) // 2
    num2 = length - len(title) - 2 - 2 * num
    output += "*" + " " * num + title + " " * (num + num2) + "*\n"
    output += "*" * length + "\n\n"
    lines = ""
    for item, data in enumerate(fields):
        output += f"{fields[item]:{lengths[item]}} "
        lines += "=" * lengths[item] + " "
    output += f"\n{lines}"
    return output


def report_header(exclude_params):
    print(TextColor.HEADER + report_header_content(title) + TextColor.ENDC)


def report_line(record, agg):
    accuracy = record[5]
    expected = record[10]
    if accuracy < expected:
        agg["worse"] += 1
        sign = "-"
    elif accuracy > expected:
        agg["better"] += 1
        sign = "+"
    else:
        agg["equal"] += 1
        sign = "="
    model = record[3]
    agg[model] += 1
    output = (
        f"{record[0]:%Y-%m-%d} {str(record[1]):>8s} {record[2]:10s} "
        f"{model:10s} {record[4]:30s} "
        f"{record[6]:3d} {record[7]:3d} {accuracy:8.7f} {expected:8.7f} "
        f"{sign}"
    )
    if not exclude_parameters:
        output += f" {record[8]}"
    return output


def report_footer(agg):
    print(TextColor.GREEN + f"we have better results {agg['better']:2d} times")
    print(TextColor.RED + f"we have worse  results {agg['worse']:2d} times")
    print(
        TextColor.MAGENTA + f"we have equal  results {agg['equal']:2d} times"
    )
    color = TextColor.LINE1
    for item in ["stree", "bagging", "adaBoost", "odte"]:
        print(color + f"{item:10s} used {agg[item]:2d} times")
        color = (
            TextColor.LINE2 if color == TextColor.LINE1 else TextColor.LINE1
        )


(
    classifier,
    exclude_parameters,
) = parse_arguments()
dbh = MySQL()
database = dbh.get_connection()
dt = Datasets(False, False, "tanveer")
title = "Best Hyperparameters found for datasets"
lengths = (10, 8, 10, 10, 30, 3, 3, 9, 11)
fields = (
    "Date",
    "Time",
    "Type",
    "Classifier",
    "Dataset",
    "Nor",
    "Std",
    "Accuracy",
    "Reference",
)
if not exclude_parameters:
    fields += ("Parameters",)
    lengths += (30,)
report_header(title)
color = TextColor.LINE1
agg = {}
for item in [
    "equal",
    "better",
    "worse",
] + models:
    agg[item] = 0
for dataset in dt:
    record = dbh.find_best(dataset[0], classifier)
    if record is None:
        print(TextColor.FAIL + f"*No results found for {dataset[0]}")
    else:
        color = (
            TextColor.LINE2 if color == TextColor.LINE1 else TextColor.LINE1
        )
        print(color + report_line(record, agg))
report_footer(agg)
dbh.close()
