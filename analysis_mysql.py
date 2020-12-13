from experimentation.Sets import Datasets
from experimentation.Utils import TextColor
from experimentation.Database import MySQL

models = ["stree", "odte", "adaBoost", "bagging"]
title = "Best model results"
lengths = (30, 9, 11, 11, 11, 11)


def report_header_content(title):
    length = sum(lengths) + len(lengths) - 1
    output = "\n" + "*" * length + "\n"
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


def report_line(line):
    output = f"{line['dataset']:{lengths[0] + 5}s} "
    data = models.copy()
    data.insert(0, "reference")
    for key, model in enumerate(data):
        output += f"{line[model]:{lengths[key + 1]}s} "
    return output


def report_footer(agg):
    print(
        TextColor.GREEN
        + f"we have better results {agg['better']['items']:2d} times"
    )
    print(
        TextColor.RED
        + f"we have worse  results {agg['worse']['items']:2d} times"
    )
    color = TextColor.LINE1
    for item in models:
        print(
            color + f"{item:10s} used {agg[item]['items']:2d} times ", end=""
        )
        print(
            color + f"better {agg[item]['better']:2d} times ",
            end="",
        )
        print(color + f"worse {agg[item]['worse']:2d} times ")
        color = (
            TextColor.LINE2 if color == TextColor.LINE1 else TextColor.LINE1
        )


dbh = MySQL()
database = dbh.get_connection()
dt = Datasets(False, False, "tanveer")
fields = ("Dataset", "Reference")
for model in models:
    fields += (f"{model}",)
report_header(title)
color = TextColor.LINE1
agg = {}
for item in [
    "better",
    "worse",
] + models:
    agg[item] = {}
    agg[item]["items"] = 0
    agg[item]["better"] = 0
    agg[item]["worse"] = 0
for dataset in dt:
    find_one = False
    line = {"dataset": color + dataset[0]}
    record = dbh.find_best(dataset[0], "any")
    max_accuracy = 0.0 if record is None else record[5]
    for model in models:
        record = dbh.find_best(dataset[0], model)
        if record is None:
            line[model] = color + "-" * 9 + "  "
        else:
            reference = record[10]
            accuracy = record[5]
            find_one = True
            agg[model]["items"] += 1
            if accuracy > reference:
                sign = "+"
                agg["better"]["items"] += 1
                agg[model]["better"] += 1
            else:
                sign = "-"
                agg["worse"]["items"] += 1
                agg[model]["worse"] += 1
            item = f"{accuracy:9.7} {sign}"
            line["reference"] = f"{reference:9.7}"
            line[model] = (
                TextColor.GREEN + TextColor.BOLD + item + TextColor.ENDC
                if accuracy == max_accuracy
                else color + item
            )
    if not find_one:
        print(TextColor.FAIL + f"*No results found for {dataset[0]}")
    else:
        color = (
            TextColor.LINE2 if color == TextColor.LINE1 else TextColor.LINE1
        )
        print(report_line(line))
report_footer(agg)
dbh.close()
