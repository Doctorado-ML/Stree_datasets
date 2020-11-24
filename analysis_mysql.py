from experimentation.Sets import Datasets
from experimentation.Utils import TextColor, MySQL

models = ["stree", "odte", "adaBoost", "bagging"]
title = "Best model results"
lengths = (30, 9, 11, 11, 11, 11)


def find_best(dataset, classifier):
    cursor = database.cursor(buffered=True)
    if classifier == "any":
        command = (
            f"select * from results r inner join reference e on "
            f"r.dataset=e.dataset where r.dataset='{dataset}' "
        )
    else:
        command = (
            f"select * from results r inner join reference e on "
            f"r.dataset=e.dataset where r.dataset='{dataset}' and classifier"
            f"='{classifier}'"
        )
    command += (
        " order by r.dataset, accuracy desc, classifier desc, type, date, time"
    )
    cursor.execute(command)
    return cursor.fetchone()


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
    print(TextColor.GREEN + f"we have better results {agg['better']:2d} times")
    print(TextColor.RED + f"we have worse  results {agg['worse']:2d} times")
    color = TextColor.LINE1
    for item in models:
        print(color + f"{item:10s} used {agg[item]:2d} times")
        color = (
            TextColor.LINE2 if color == TextColor.LINE1 else TextColor.LINE1
        )


database = MySQL.get_connection()
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
    agg[item] = 0
for dataset in dt:
    find_one = False
    line = {"dataset": color + dataset[0]}
    record = find_best(dataset[0], "any")
    max_accuracy = 0.0 if record is None else record[5]
    for model in models:
        record = find_best(dataset[0], model)
        if record is None:
            line[model] = color + "-" * 9 + "  "
        else:
            reference = record[10]
            accuracy = record[5]
            find_one = True
            agg[model] += 1
            if accuracy > reference:
                sign = "+"
                agg["better"] += 1
            else:
                sign = "-"
                agg["worse"] += 1
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
