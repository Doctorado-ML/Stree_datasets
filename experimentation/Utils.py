import os
import mysql.connector


class TextColor:
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    MAGENTA = "\033[95m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    HEADER = MAGENTA
    LINE1 = BLUE
    LINE2 = CYAN
    SUCCESS = GREEN
    WARNING = YELLOW
    FAIL = RED
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class MySQL:
    @staticmethod
    def get_connection():
        config = dict()
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, ".myconfig")) as f:
            for line in f.read().splitlines():
                key, value = line.split("=")
                config[key] = value
        return mysql.connector.connect(**config)
