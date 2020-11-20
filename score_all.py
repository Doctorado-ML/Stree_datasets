import sys
import time
import warnings

from experimentation.Sets import Datasets
from odte import Odte
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from stree import Stree

classifiers = {
    "Stree ": Stree(C=256, max_iter=1e5, random_state=1),
    "D.Tree": DecisionTreeClassifier(),
    "SVC(L)": LinearSVC(),
    "SVC(R)": SVC(kernel="rbf"),
    "SVC(P)": SVC(kernel="poly"),
    "R.For.": RandomForestClassifier(),
    "Odte  ": Odte(random_state=1, n_jobs=-1),
    # "Oc2   ": ObliqueTree()
}


def header():
    print(f"Score {dtype} files")
    initial = f"{'Dataset':30s}"
    sec_line = "=" * 30
    for classifier in classifiers:
        initial += " Time(s)  " + classifier
        sec_line += " ======== " + "=" * len(classifier)
    print(initial)
    print(sec_line)


warnings.filterwarnings("ignore")
dtype = "all"
if len(sys.argv) > 1:
    set_name = sys.argv[1]
    if len(sys.argv) > 2:
        dtype = sys.argv[2]
else:
    set_name = "aaai"
if set_name != "aaai" and set_name != "tanveer":
    print("First parameter has to be one of: {aaai, tanveer}")
    exit(1)
datasets = Datasets(False, False, set_name)
clf = None
header()
better = worse = equal = 0
for dataset in datasets:
    if dataset[1] == dtype or dtype == "all" or dtype == "any":
        X, y = datasets.load(dataset[0])
        output = ""
        odte_score = stree_score = 0.0
        for model_name, clf in classifiers.items():
            now = time.time()
            clf.set_params(random_state=0)
            score = clf.fit(X, y).score(X, y)
            if model_name.strip() == "Stree":
                stree_score = score
            if model_name.strip() == "Odte":
                odte_score = score
            output += f" {time.time()-now:-7.2f} {score:.4f} "
        line = f"{dataset[0]:30s} " + output
        if stree_score > odte_score:
            line += "-"
            worse += 1
        elif stree_score < odte_score:
            line += "+"
            better += 1
        else:
            line += "="
            equal += 1
        print(line)
print(f"Odte is better than Stree {better:2d} times")
print(f"Odte is worse  than Stree {worse:2d} times")
print(f"Odte is equal  to   Stree {equal:2d} times")
print(clf.get_params() if clf is not None else "No dataset match criteria")
