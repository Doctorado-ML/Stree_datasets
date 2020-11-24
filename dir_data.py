import os
import pandas as pd
import numpy as np
from experimentation.Utils import TextColor
from experimentation.Sets import Datasets

path = os.path.join(os.getcwd(), "data/tanveer")
color = TextColor.LINE1
dt = np.array(list(Datasets(False, False, "tanveer")), dtype="object")
dt = dt[:, 0]
good = bad = 0
for folder in sorted(os.listdir(path)):
    file_name = os.path.join(path, folder, f"{folder}_R.dat")
    try:
        data = pd.read_csv(
            file_name,
            sep="\t",
            index_col=0,
        )
        X = data.drop("clase", axis=1).to_numpy()
        y = data["clase"].to_numpy()
        sign = "*" if folder in dt else "-"
        print(color + f"{folder:30s} {str(X.shape):>10s} {sign}")
        color = (
            TextColor.LINE1 if color == TextColor.LINE2 else TextColor.LINE2
        )
        good += 1
    except FileNotFoundError:
        print(TextColor.FAIL + f"{folder} not found.")
        bad += 1
print(TextColor.SUCCESS + f"{good:3d} datasets Ok.")
print(TextColor.FAIL + f"{bad:3d} datasets Wrong.")
