import os
import time
import numpy as np
import pandas as pd
from scipy.io import arff
from stree import Stree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

folder = (
    "/Volumes/Datos/OneDrive - Universidad de Castilla-La Mancha/"
    "Doctorado2019/Compartida/FuentesDescargados/data-4/"
)

name = "yeast"
random_state = 1

file_name = os.path.join(folder, name, f"{name}.arff")
data, meta = arff.loadarff(file_name)
df = pd.DataFrame(data)
y = df["clase"].to_numpy().astype(np.int16)
df.drop(columns="clase", inplace=True)
X = df.to_numpy().astype(np.float16)
print(f"Xshape {X.shape} Xtype {X.dtype}")
print(f"yshape {y.shape} ytype {y.dtype}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=random_state
)

clf = Stree(
    random_state=random_state,
    C=1e5,
    max_iter=1e5,
    kernel="poly",
    degree=5,
    gamma=0.8,
)
now = time.time()
scores = cross_val_score(clf, X, y, cv=5)
print(f"Accuracy for {name}: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
print(f"Took : {time.time() - now:.2f} seconds")
print(f"Score one tree all samples .: {clf.fit(X, y).score(X, y):.4f}")
print(
    f"Score one tree train/test .: "
    f"{clf.fit(X_train, y_train).score(X_test, y_test):.4f}"
)
print("*" * 80)
