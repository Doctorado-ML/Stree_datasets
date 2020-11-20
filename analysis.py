import sweetviz as sv
import pandas as pd
import numpy as np
from experimentation import Dataset


def dataframe(X, y):
    label = y.reshape(-1, 1)
    return pd.DataFrame(np.concatenate((X, label), axis=1))


# datasets = Dataset()
# for dataset in datasets:
#     if dataset[0] > "usps" and dataset[0] != "usps":
#         data = dataframe(*datasets.load(dataset[0]))
#         print(dataset[0], data.shape)
#         report = sv.analyze(source=[data, dataset[0]])
#         report.show_html(f"html/{dataset[0]}.html")

# datasets = Dataset()
# for dataset in ['mnist', 'protein', 'usps', 'shuttle']:
#     data = dataframe(*datasets.load(dataset))
#     print(dataset, data.shape)
#     report = sv.analyze(source=[data, dataset], pairwise_analysis="off")
#     report.show_html(f"html/{dataset}.html")

datasets = Dataset(normalize=True)
for dataset in ["shuttle"]:
    data = dataframe(*datasets.load(dataset))
    print(dataset, data.shape)
    report = sv.analyze(source=[data, dataset], pairwise_analysis="off")
    report.show_html(f"html/{dataset}.html")
