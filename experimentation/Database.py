import os
import sqlite3
from datetime import datetime
from abc import ABC
from typing import List
import mysql.connector
from ast import literal_eval as make_tuple
from sshtunnel import SSHTunnelForwarder
from .Models import ModelBase
from .Utils import TextColor


class MySQL:
    def __init__(self):
        self._server = None
        self._tunnel = False
        self._config_db = dict()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, ".myconfig")) as f:
            for line in f.read().splitlines():
                key, value = line.split("=")
                self._config_db[key] = value
        config_tunnel = dict()
        with open(os.path.join(dir_path, ".tunnel")) as f:
            for line in f.read().splitlines():
                key, value = line.split("=")
                config_tunnel[key] = value
        config_tunnel["remote_bind_address"] = make_tuple(
            config_tunnel["remote_bind_address"]
        )
        config_tunnel["ssh_address_or_host"] = make_tuple(
            config_tunnel["ssh_address_or_host"]
        )
        self._tunnel = config_tunnel["enabled"] == "1"
        if self._tunnel:
            del config_tunnel["enabled"]
            self._server = SSHTunnelForwarder(**config_tunnel)
            self._server.daemon_forward_servers = True

    def get_connection(self):
        if self._tunnel:
            self._server.start()
            self._config_db["port"] = self._server.local_bind_port
        self._database = mysql.connector.connect(**self._config_db)
        return self._database

    def find_best(self, dataset, classifier="any"):
        cursor = self._database.cursor(buffered=True)
        if classifier == "any":
            command = (
                f"select * from results r inner join reference e on "
                f"r.dataset=e.dataset where r.dataset='{dataset}' "
            )
        else:
            command = (
                f"select * from results r inner join reference e on "
                f"r.dataset=e.dataset where r.dataset='{dataset}' and "
                f"classifier='{classifier}'"
            )
        command += (
            " order by r.dataset, accuracy desc, classifier desc, "
            "type, date, time"
        )
        cursor.execute(command)
        return cursor.fetchone()

    def close(self):
        if self._tunnel:
            self._server.close()


class BD(ABC):
    _folder = "./data/results"
    _con = None

    def __init__(self, host: str, model: ModelBase) -> None:
        self._model = model
        self._host = host
        self._database = os.path.join(
            self._folder, f"{host}_{model}_experiments.sqlite3"
        )
        self._con = sqlite3.connect(self._database)
        # return dict as a result of select
        self._con.row_factory = sqlite3.Row
        self._check_build()
        # accumulators used in reports
        self._best = self._worse = self._equal = 0

    def _check_build(self) -> None:
        """Check if the tables are created and create them if they don't"""
        commands = [
            'create table if not exists "outcomes" ("dataset" varchar NOT NULL'
            ',"date" datetime NOT NULL DEFAULT NULL,"fit_time" num NOT NULL '
            'DEFAULT NULL, "fit_time_std" num, "score_time" num NOT NULL '
            'DEFAULT NULL, "score_time_std" num, "train_score" num NOT NULL '
            'DEFAULT NULL, "train_score_std" num, "test_score" num NOT NULL '
            'DEFAULT NULL, "test_score_std" num, "parameters" text DEFAULT '
            'NULL, "normalize" int NOT NULL DEFAULT 0, "standardize" int NOT '
            "NULL DEFAULT 0, PRIMARY KEY (dataset, date));",
            'create table if not exists "hyperparameters" ("dataset" varchar '
            'NOT NULL,"date" datetime NOT NULL DEFAULT NULL,"fit_time" num NOT'
            ' NULL DEFAULT NULL, "fit_time_std" num, "score_time" num NOT NULL'
            ' DEFAULT NULL, "score_time_std" num, "train_score" num NOT NULL '
            'DEFAULT NULL, "train_score_std" num, "test_score" num NOT NULL '
            'DEFAULT NULL, "test_score_std" num, "parameters" text DEFAULT '
            'NULL, "normalize" int NOT NULL DEFAULT 0, "standardize" int NOT '
            "NULL DEFAULT 0, PRIMARY KEY (dataset, normalize, standardize));",
            'create table if not exists "reference" ("dataset" varchar NOT '
            'NULL,"score" num NOT NULL, PRIMARY KEY (dataset));',
            "INSERT or replace INTO reference (dataset, score) VALUES "
            "('balance-scale', '0.904628'),"
            "('balloons', '0.6625'),"
            "('breast-cancer-wisc-diag', '0.974345'),"
            "('breast-cancer-wisc-prog', '0.79934'),"
            "('breast-cancer-wisc', '0.970256'),"
            "('breast-cancer', '0.73824'),"
            "('cardiotocography-10clases', '0.827761'),"
            "('cardiotocography-3clases', '0.920134'),"
            "('conn-bench-sonar-mines-rocks', '0.833654'),"
            "('cylinder-bands', '0.769141'),"
            "('dermatology', '0.973278'),"
            "('echocardiogram', '0.848527'),"
            "('fertility', '0.884'),"
            "('haberman-survival', '0.739254'),"
            "('heart-hungarian', '0.820475'),"
            "('hepatitis', '0.823203'),"
            "('ilpd-indian-liver', '0.715028'),"
            "('ionosphere', '0.944215'),"
            "('iris', '0.978656'),"
            "('led-display', '0.7102'),"
            "('libras', '0.891111'),"
            "('low-res-spect', '0.90282'),"
            "('lymphography', '0.855405'),"
            "('mammographic', '0.827472'),"
            "('molec-biol-promoter', '0.818269'),"
            "('musk-1', '0.876471'),"
            "('oocytes_merluccius_nucleus_4d', '0.839963'),"
            "('oocytes_merluccius_states_2f', '0.929963'),"
            "('oocytes_trisopterus_nucleus_2f', '0.833333'),"
            "('oocytes_trisopterus_states_5b', '0.931579'),"
            "('parkinsons', '0.920221'),"
            "('pima', '0.767188'),"
            "('pittsburg-bridges-MATERIAL', '0.864286'),"
            "('pittsburg-bridges-REL-L', '0.695929'),"
            "('pittsburg-bridges-SPAN', '0.68913'),"
            "('pittsburg-bridges-T-OR-D', '0.87437'),"
            "('planning', '0.725579'),"
            "('post-operative', '0.711742'),"
            "('seeds', '0.956303'),"
            "('statlog-australian-credit', '0.678281'),"
            "('statlog-german-credit', '0.7562'),"
            "('statlog-heart', '0.842299'),"
            "('statlog-image', '0.976194'),"
            "('statlog-vehicle', '0.800673'),"
            "('synthetic-control', '0.990333'),"
            "('tic-tac-toe', '0.985385'),"
            "('vertebral-column-2clases', '0.849153'),"
            "('wine', '0.993281'),"
            "('zoo', '0.960385')",
        ]
        for command in commands:
            self.execute(command)

    def mirror(
        self, exp_type, dataset, normalize, standardize, accuracy, parameters
    ) -> None:
        """Create a record in MySQL database

        :param record: data to insert in database
        :type record: dict
        """
        dbh = MySQL()
        database = dbh.get_connection()
        command_insert = (
            "replace into results (date, time, type, accuracy, "
            "dataset, classifier, norm, stand, parameters) values (%s, %s, "
            "%s, %s, %s, %s, %s, %s, %s)"
        )
        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")
        values = (
            date,
            time,
            exp_type,
            accuracy,
            dataset,
            self._model,
            normalize,
            standardize,
            parameters,
        )
        cursor = database.cursor()
        cursor.execute(command_insert, values)
        database.commit()
        dbh.close()

    def execute(self, command: str) -> None:
        c = self._con.cursor()
        c.execute(command)
        c.close()
        self._con.commit()

    def header(
        self,
        title: str,
        lengths: List[int],
        fields: List[str],
        exclude_params,
    ) -> str:
        length = 148 if exclude_params else 170
        title += f" -- {self._model} in {self._host} --"
        output = "\n" + "*" * length + "\n"
        num = (length - len(title) - 2) // 2
        num2 = length - len(title) - 2 - 2 * num
        output += "*" + " " * num + title + " " * (num + num2) + "*\n"
        output += "*" * length + "\n\n"
        for field, length in zip(fields, lengths):
            output += ("{0:" + str(length) + "} ").format(field)
        output += "\n"
        for length in lengths:
            output += "=" * length + " "
        return output

    def check_result(self, test, reference) -> str:
        if test > reference:
            self._best += 1
            result = "+"
        elif test < reference:
            self._worse += 1
            result = "-"
        else:
            self._equal += 1
            result = "="
        return result

    def report_line(self, data, exclude_params):
        data = list(data)
        dataset = data.pop(0)
        reference = data.pop()
        _ = data.pop()  # remove dataset name of inner join
        exec_date = data.pop(0)
        standardize = data.pop()
        normalize = data.pop()
        parameters = data.pop()
        result = self.check_result(data[6], reference)
        if exclude_params:
            parameters = ""
        output = ""
        index = 0
        for item in data:
            if index % 2:
                fact = f" (+/- {item * 2:6.2f}) "
            else:
                if index > 3:
                    fact = f"{item:7.4f}"
                else:
                    fact = f"{item:7.2f}"
            output += fact
            index += 1
        return (
            f"{dataset:30s} {exec_date:10s} {normalize} {standardize} {output}"
            f"{reference:1.5f} {result} {parameters}"
        )

    def report_header(self, title, exclude_params):
        lengths = [30, 19, 1, 1, 20, 20, 20, 20, 9, 21]
        fields = [
            "Dataset",
            "Date",
            "N",
            "S",
            "Fit Time (sec)",
            "Score Time (sec)",
            "Score on Train",
            "Score on Test",
            "Reference",
            "Parameters",
        ]
        if exclude_params:
            fields.pop()
            lengths.pop()
        return self.header(title, lengths, fields, exclude_params)

    def report_footer(self):
        print(
            TextColor.GREEN
            + f"{self._model} has better results {self._best:2d} times"
        )
        print(
            TextColor.RED
            + f"{self._model} has worse  results {self._worse:2d} times"
        )
        print(
            TextColor.MAGENTA
            + f"{self._model} has equal  results {self._equal:2d} times"
        )


class Outcomes(BD):
    def __init__(self, host: str, model):
        self._table = "outcomes"
        super().__init__(host=host, model=model)

    def store(self, dataset, normalize, standardize, parameters, results):
        outcomes = ["fit_time", "score_time", "train_score", "test_score"]
        data = ""
        for index in outcomes:
            data += ", " + str(results[index].mean()) + ", "
            data += str(results[index].std())
        command = (
            f"insert or replace into {self._table} ('dataset', 'parameters', "
            "'date', 'normalize', 'standardize'"
        )
        for field in outcomes:
            command += f",'{field}', '{field}_std'"
        command += f") values('{dataset}', '{parameters}', DateTime('now', "
        command += f"'localtime'), '{int(normalize)}', '{int(standardize)}'"
        command += data + ")"
        command = command.replace("nan", "null")
        self.execute(command)
        self.mirror(
            "crossval",
            dataset,
            normalize,
            standardize,
            float(results["test_score"].mean()),
            parameters,
        )

    def report(self, dataset, exclude_params):
        cursor = self._con.cursor()
        suffix = "" if dataset == "all" else f"WHERE dataset='{dataset}'"
        cursor.execute(
            f"SELECT * FROM {self._table} o {suffix} inner join reference r on"
            " o.dataset=r.dataset order by dataset, date desc;"
        )
        records = cursor.fetchall()
        num_records = len(records)
        title = f"5 Folds Cross Validation: {dataset} - {num_records} records"
        print(
            TextColor.HEADER
            + self.report_header(title, exclude_params)
            + TextColor.ENDC
        )
        color = TextColor.LINE2
        for record in records:
            color = (
                TextColor.LINE1
                if color == TextColor.LINE2
                else TextColor.LINE2
            )
            print(
                color
                + self.report_line(record, exclude_params)
                + TextColor.ENDC
            )
        if records == []:
            print(
                TextColor.WARNING
                + "           No records yet"
                + TextColor.ENDC
            )
        else:
            self.report_footer()
        cursor.close()


class Hyperparameters(BD):
    def __init__(self, host: str, model):
        self._table = "hyperparameters"
        super().__init__(host=host, model=model)

    def store(
        self,
        dataset,
        time,
        grid,
        parameters,
        normalize,
        standardize,
        grid_type,
    ):
        rosetta = [
            ("mean_fit_time", "fit_time"),
            ("std_fit_time", "fit_time_std"),
            ("mean_score_time", "score_time"),
            ("std_score_time", "score_time_std"),
            ("mean_test_score", "test_score"),
            ("std_test_score", "test_score_std"),
            ("mean_train_score", "train_score"),
            ("std_train_score", "train_score_std"),
            ("params", "parameters"),
        ]
        # load outcomes vector
        outcomes = {}
        for item, bd_item in rosetta:
            outcomes[bd_item] = grid.cv_results_[item][grid.best_index_]
        outcomes["parameters"] = parameters
        outcomes["normalize"] = int(normalize)
        outcomes["standardize"] = int(standardize)
        rosetta.append(("_", "normalize"))
        rosetta.append(("_", "standardize"))
        command = f"insert or replace into {self._table} ('dataset', 'date'"
        command_values = f"values ('{dataset}', DateTime('now', 'localtime')"
        for _, item in rosetta:
            command += f", '{item}'"
            command_values += (
                f", {outcomes[item]}"
                if item != "parameters"
                else f", '{outcomes[item]}'"
            )
        command += ") "
        command_values += ")"
        self.execute(command + command_values)
        accuracy = float(outcomes["test_score"])
        self.mirror(
            grid_type, dataset, normalize, standardize, accuracy, parameters
        )

    def report(self, dataset, exclude_params):
        cursor = self._con.cursor()
        cursor.execute(
            f"SELECT * FROM {self._table} h inner join reference r on "
            "r.dataset=h.dataset order by dataset, date desc;"
        )
        records = cursor.fetchall()
        num_records = len(records)
        title = f"Grid Searches done so far - {num_records} records"
        print(
            TextColor.HEADER
            + self.report_header(title, exclude_params)
            + TextColor.ENDC
        )
        color = TextColor.LINE2
        for record in records:
            color = (
                TextColor.LINE1
                if color == TextColor.LINE2
                else TextColor.LINE2
            )
            print(
                color
                + self.report_line(record, exclude_params)
                + TextColor.ENDC
            )
        cursor.close()
        self.report_footer()

    def get_params(self, dataset):
        cursor = self._con.cursor()
        cursor.execute(
            f"SELECT parameters, normalize, standardize FROM {self._table} "
            f"where dataset='{dataset}' order by test_score desc;"
        )
        record = cursor.fetchone()
        if record is None:
            raise ValueError(f"parameters not found for dataset {dataset}")
        return record["parameters"], record["normalize"], record["standardize"]
