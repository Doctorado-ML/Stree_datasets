import sys
from experimentation.Sets import Datasets

set_name = "aaai"
if len(sys.argv) > 1:
    set_name = sys.argv[1]
if set_name != "aaai" and set_name != "tanveer":
    print("First parameter has to be one of: {aaai, tanveer}")
    exit(1)
datasets = Datasets(False, False, set_name)
datasets.report()
