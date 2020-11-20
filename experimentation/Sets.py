from __future__ import annotations
import os
from typing import Tuple, List
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from .Utils import TextColor

tsets = List[Tuple[str, str, str, Tuple[int, int]]]
tdataset = Tuple[np.array, np.array]


class Diterator:
    def __init__(self, data: tsets):
        self._stack: tsets = data.copy()

    def __next__(self):
        if len(self._stack) == 0:
            raise StopIteration()
        return self._stack.pop(0)


class Dataset_Base:
    def __init__(self, normalize: bool, standardize: bool) -> None:
        self._data_folder = "data"
        self._normalize = normalize
        self._standardize = standardize

    def load(self, name: str) -> tdataset:
        """Datasets have to implement this

        :param name: dataset name
        :type name: str
        :return: X, y np.arrays
        :rtype: tdataset
        """
        pass

    def normalize(self, data: np.array) -> np.array:
        min_data = data.min()
        return (data - min_data) / (data.max() - min_data)

    def standardize(self, data: np.array) -> np.array:
        return (data - data.mean()) / data.std()

    def get_params(self) -> str:
        return f"normalize={self._normalize}, standardize={self._standardize}"

    def post_process(self, X: np.array, y: np.array) -> tdataset:
        if self._standardize and self._normalize:
            X = self.standardize(self.normalize(X))
        elif self._standardize:
            X = self.standardize(X)
        elif self._normalize:
            X = self.normalize(X)
        return X, y

    def __iter__(self) -> Diterator:
        return Diterator(self.data_sets)

    def __str__(self) -> str:
        out = ""
        for dataset in self.data_sets:
            out += f" {dataset[0]},"
        return out

    def report(self) -> None:
        print(
            TextColor.HEADER
            + "(#) Dataset                       Samples  Feat. #Cl. y typ "
            + "X type  f_type"
        )
        print(
            "=== ============================= ======== ===== ==== ===== "
            "======= ======" + TextColor.ENDC
        )
        color = TextColor.LINE2
        for number, dataset in enumerate(self.data_sets):
            X, y = self.load(dataset[0])  # type: ignore
            samples, features = X.shape
            classes = len(np.unique(y))
            color = (
                TextColor.LINE1
                if color == TextColor.LINE2
                else TextColor.LINE2
            )
            print(
                color + f"#{number + 1:02d} {dataset[0]:30s} {samples:7,d} "
                f"{features:5d} {classes:4d} {y.dtype} {str(X.dtype):7s} "
                f"{dataset[1]}" + TextColor.ENDC
            )
            # Check dataset is Ok
            # data type
            if str(X.dtype) != dataset[2]:
                raise ValueError(
                    f"dataset {dataset[0]} has wrong data type. "
                    f"It shoud have {dataset[2]} but has {X.dtype}"
                )
            # dimensions
            if X.shape != dataset[3]:
                raise ValueError(
                    f"dataset {dataset[0]} has wrong X shape. "
                    f"It shoud have {dataset[3]} but has {X.shape}"
                )
            if y.shape != (X.shape[0],):
                raise ValueError(
                    f"dataset {dataset[0]} has wrong y shape. "
                    f"It shoud have {(X.shape[0],)} but has {y.shape}"
                )

        print(
            TextColor.SUCCESS
            + "* All Data types and shapes are Ok."
            + TextColor.ENDC
        )


class Datasets_Tanveer(Dataset_Base):
    def __init__(
        self, normalize: bool = False, standardize: bool = False
    ) -> None:
        super().__init__(normalize, standardize)
        self._folder = os.path.join(self._data_folder, "tanveer")
        self.data_sets: tsets = [
            # (name), (filetype), (sampes type)
            ("balance-scale", "Rdat", "float64", (625, 4)),
            ("balloons", "Rdat", "float64", (16, 4)),
            ("breast-cancer-wisc-diag", "Rdat", "float64", (569, 30)),
            ("breast-cancer-wisc-prog", "Rdat", "float64", (198, 33)),
            ("breast-cancer-wisc", "Rdat", "float64", (699, 9)),
            ("breast-cancer", "Rdat", "float64", (286, 9)),
            ("cardiotocography-10clases", "Rdat", "float64", (2126, 21)),
            ("cardiotocography-3clases", "Rdat", "float64", (2126, 21)),
            ("conn-bench-sonar-mines-rocks", "Rdat", "float64", (208, 60)),
            ("cylinder-bands", "Rdat", "float64", (512, 35)),
            ("dermatology", "Rdat", "float64", (366, 34)),
            ("echocardiogram", "Rdat", "float64", (131, 10)),
            ("fertility", "Rdat", "float64", (100, 9)),
            ("haberman-survival", "Rdat", "float64", (306, 3)),
            ("heart-hungarian", "Rdat", "float64", (294, 12)),
            ("hepatitis", "Rdat", "float64", (155, 19)),
            ("ilpd-indian-liver", "Rdat", "float64", (583, 9)),
            ("ionosphere", "Rdat", "float64", (351, 33)),
            ("iris", "Rdat", "float64", (150, 4)),
            ("led-display", "Rdat", "float64", (1000, 7)),
            ("libras", "Rdat", "float64", (360, 90)),
            ("low-res-spect", "Rdat", "float64", (531, 100)),
            ("lymphography", "Rdat", "float64", (148, 18)),
            ("mammographic", "Rdat", "float64", (961, 5)),
            ("molec-biol-promoter", "Rdat", "float64", (106, 57)),
            ("musk-1", "Rdat", "float64", (476, 166)),
            ("oocytes_merluccius_nucleus_4d", "Rdat", "float64", (1022, 41)),
            ("oocytes_merluccius_states_2f", "Rdat", "float64", (1022, 25)),
            ("oocytes_trisopterus_nucleus_2f", "Rdat", "float64", (912, 25)),
            ("oocytes_trisopterus_states_5b", "Rdat", "float64", (912, 32)),
            ("parkinsons", "Rdat", "float64", (195, 22)),
            ("pima", "Rdat", "float64", (768, 8)),
            ("pittsburg-bridges-MATERIAL", "Rdat", "float64", (106, 7)),
            ("pittsburg-bridges-REL-L", "Rdat", "float64", (103, 7)),
            ("pittsburg-bridges-SPAN", "Rdat", "float64", (92, 7)),
            ("pittsburg-bridges-T-OR-D", "Rdat", "float64", (102, 7)),
            ("planning", "Rdat", "float64", (182, 12)),
            ("post-operative", "Rdat", "float64", (90, 8)),
            ("seeds", "Rdat", "float64", (210, 7)),
            ("statlog-australian-credit", "Rdat", "float64", (690, 14)),
            ("statlog-german-credit", "Rdat", "float64", (1000, 24)),
            ("statlog-heart", "Rdat", "float64", (270, 13)),
            ("statlog-image", "Rdat", "float64", (2310, 18)),
            ("statlog-vehicle", "Rdat", "float64", (846, 18)),
            ("synthetic-control", "Rdat", "float64", (600, 60)),
            ("tic-tac-toe", "Rdat", "float64", (958, 9)),
            ("vertebral-column-2clases", "Rdat", "float64", (310, 6)),
            ("wine", "Rdat", "float64", (178, 13)),
            ("zoo", "Rdat", "float64", (101, 16)),
        ]

    def load(self, name: str) -> tdataset:
        data = pd.read_csv(
            os.path.join(self._folder, name, f"{name}_R.dat"),
            sep="\t",
            index_col=0,
        )
        X = data.drop("clase", axis=1).to_numpy()
        y = data["clase"].to_numpy()
        return X, y


class Datasets_AAAI(Dataset_Base):
    def __init__(
        self, normalize: bool = False, standardize: bool = False
    ) -> None:
        super().__init__(normalize, standardize)
        self._folder: str = os.path.join(self._data_folder, "aaai")
        self.data_sets: tsets = [
            # (name), (filetype), (sampes type)
            ("breast", "csv", "int16", (683, 9)),
            ("cardiotoc", "csv", "int16", (2126, 41)),
            ("cod-rna", "sparse", "float16", (331152, 8)),
            ("connect4", "sparse", "int16", (67557, 126)),
            ("covtype", "npz", "int16", (581012, 54)),
            ("diabetes", "csv", "float16", (768, 8)),
            ("dna", "csv", "float16", (3186, 180)),
            ("fourclass", "sparse", "int16", (862, 2)),
            ("glass", "csv", "float16", (214, 9)),
            ("heart", "csv", "float16", (270, 13)),
            ("ijcnn1", "sparse", "float16", (141691, 22)),
            ("iris", "csv", "float16", (150, 4)),
            ("letter", "npz", "int16", (20000, 16)),
            ("mnist", "npy", "int16", (70000, 784)),
            ("pendigits", "npy", "int16", (10992, 16)),
            ("protein", "sparse", "float16", (24387, 357)),
            ("satimage", "npy", "int16", (6435, 36)),
            ("segment", "sparse", "float16", (2310, 19)),
            ("shuttle", "npy", "int16", (58000, 9)),
            ("usps", "npz", "float16", (9298, 256)),
            ("vehicle", "sparse", "float16", (846, 18)),
            ("wine", "csv", "float16", (178, 13)),
        ]

    def load_dataset(
        self, name: str, file_type: str, data_type: str
    ) -> tdataset:
        return getattr(self, f"load_{file_type}_dataset")(name, data_type)

    def load(self, name: str) -> tdataset:
        for dataset in self.data_sets:
            if name == dataset[0]:
                return self.post_process(
                    *self.load_dataset(*dataset[:3])  # type: ignore
                )
        raise ValueError(
            f"{name} is not a valid dataset, has to be one of {str(self)}"
        )

    def load_csv_dataset(self, name: str, dtype: str) -> tdataset:
        data = np.genfromtxt(
            os.path.join(self._folder, f"{name}.csv"),
            delimiter=",",
            dtype=dtype,
        )
        features = data.shape[1]
        return data[:, : features - 1], data[:, -1].astype(np.int16)

    def load_npy_dataset(self, name: str, _: str) -> tdataset:
        data = np.load(os.path.join(self._folder, f"{name}.npy"))
        features = data.shape[1]
        return data[:, : features - 1], data[:, -1].astype(np.int16)

    def load_npz_dataset(self, name, _):
        data = np.load(os.path.join(self._folder, f"{name}.npz"))
        return data["arr_0"], data["arr_1"]

    def load_sparse_dataset(self, name: str, _: str) -> tdataset:
        X, y = np.load(
            os.path.join(self._folder, f"{name}.npy"), allow_pickle=True
        )
        if str(X.dtype) == "float16":
            # can't do todense with np.float16
            XX = X.astype(np.float64).todense().astype(np.float16)
        else:
            XX = X.todense()
        return XX, y


class Datasets:
    def __init__(
        self,
        normalize: bool = False,
        standardize: bool = False,
        set_of_files: str = "aaai",
    ) -> None:
        self._model = (
            Datasets_AAAI(normalize, standardize)
            if set_of_files == "aaai"
            else Datasets_Tanveer(normalize, standardize)
        )

    def load(self, name: str) -> tdataset:
        return self._model.load(name)

    def post_process(self, X: np.array, y: np.array) -> tdataset:
        return self._model.post_process(X, y)

    def report(self) -> None:
        return self._model.report()

    def get_params(self) -> str:
        return self._model.get_params()

    def __iter__(self) -> Diterator:
        return Diterator(self._model.data_sets)

    def __str__(self) -> str:
        return self._model.__str__()
