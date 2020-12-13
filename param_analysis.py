import json
import argparse
import collections
from typing import Tuple
from experimentation.Database import MySQL
from experimentation.Sets import Datasets
from experimentation.Utils import TextColor

kernel_names = ["linear", "rbf", "poly"]


class Aggregation:
    def __init__(self, dbh):
        self._dbh = dbh
        self._report = {}
        self._model_names = ["stree", "adaBoost", "bagging", "odte"]
        self._kernel_names = kernel_names

    def find_values(self, dataset, parameter):
        result = []
        for data in self._report[dataset]:
            base_parameter = f"base_estimator__{parameter}"
            if parameter in data.keys():
                result.append(data[parameter])
            if base_parameter in data.keys():
                result.append(data[base_parameter])
        try:
            result_ordered = sorted(result)
            return result_ordered
        except TypeError:
            return result

    def load(self):
        dt = Datasets(False, False, "tanveer")
        print("Aggregating data of best results ...")
        for dataset in dt:
            if result := self._dbh.find_best(dataset[0]):
                accuracy = result[5]
                expected = result[10]
                model = result[3]
                json_result = json.loads(result[8])
                if "kernel" in json_result.keys():
                    kernel = json_result["kernel"]
                elif "base_estimator__kernel" in json_result.keys():
                    kernel = json_result["base_estimator__kernel"]
                else:
                    kernel = "linear"
                best = accuracy > expected
                self._report[dataset[0]] = {
                    "model": model,
                    "kernel": kernel,
                    "parameters": json_result,
                    "best": best,
                }

    @staticmethod
    def report_header(title, lengths, fields, parameter):
        length = sum(lengths) + len(lengths) - 1
        output = "\n" + "*" * length + "\n"
        title = title + f" -- {parameter} parameter --"
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

    def report(self, parameter):
        agg = {}
        agg_result = collections.OrderedDict()
        title = "Best Hyperparameters found for datasets"
        lengths = (32, 10, 7, 20)
        fields = (
            "Dataset",
            "Classifier",
            "Kernel",
            "Parameter Value",
        )
        print(Aggregation.report_header(title, lengths, fields, parameter))
        for i in self._kernel_names + self._model_names:
            agg[i] = {}
            agg[i]["total"] = 0
            agg[i]["better"] = 0
            agg[i]["worse"] = 0
        for dataset, data in self._report.items():
            kernel = data["kernel"]
            model = data["model"]
            if data["best"]:
                key = "better"
                sign = "+"
            else:
                key = "worse"
                sign = "-"
            base_parameter = f"base_estimator__{parameter}"
            result = ""
            if parameter in data["parameters"]:
                result = data["parameters"][parameter]
                try:
                    agg_result[result] += 1
                except KeyError:
                    agg_result[result] = 1
            elif base_parameter in data["parameters"]:
                result = data["parameters"][base_parameter]
                try:
                    agg_result[result] += 1
                except KeyError:
                    agg_result[result] = 1
            print(f"{sign} {dataset:30s} {model:10s} {kernel:7s} {result}")
            agg[kernel]["total"] += 1
            agg[kernel][key] += 1
            agg[model]["total"] += 1
            agg[model][key] += 1
        print(TextColor.BOLD, "Models", TextColor.ENDC)
        for i in self._model_names:
            print(
                f"{i:10} has {agg[i]['total']:2} results {agg[i]['better']:2} "
                f"better {agg[i]['worse']:2} worse"
            )
        print(TextColor.BOLD, "Kernels", TextColor.ENDC)
        for i in self._kernel_names:
            print(
                f"{i:10} has {agg[i]['total']:2} results {agg[i]['better']:2} "
                f"better {agg[i]['worse']:2} worse"
            )
        print(TextColor.BOLD, f"{parameter} Values:", TextColor.ENDC)
        try:
            max_len = f"{len(max(agg_result.keys(), key=len))}s"
        except TypeError:
            max_len = "10.2f"
        for key in sorted(agg_result):
            print(f"{key:{max_len}} -> {agg_result[key]:2d} times")


def parse_arguments() -> Tuple[str, str, str, bool, bool]:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-p",
        "--param",
        type=str,
        default="C",
    )
    args = ap.parse_args()
    return (args.param,)


(param,) = parse_arguments()
dbh = MySQL()
dbh.get_connection()
agg = Aggregation(dbh)
agg.load()
agg.report(param)
dbh.close()
