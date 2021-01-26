from stree import Stree
from typing import Union, Optional, List
from abc import ABC
from sklearn.ensemble import (
    AdaBoostClassifier,  # type: ignore
    BaggingClassifier,  # type: ignore
)
from sklearn.ensemble import BaseEnsemble  # type: ignore
from sklearn.base import BaseEstimator  # type: ignore
from sklearn.svm import LinearSVC  # type: ignore
from sklearn.tree import DecisionTreeClassifier  # type: ignore
from odte import Odte


class ModelBase(ABC):
    def __init__(self, random_state: Optional[int]):
        self._random_state = random_state

    def get_model_name(self) -> str:
        return self._model_name

    def get_model(self) -> Union[BaseEnsemble, BaseEstimator]:
        return self._clf

    def get_parameters(self) -> dict:
        return self._param_grid

    def modified_parameters(self, optimum_parameters) -> dict:
        result = dict()
        # useful for ensembles
        excluded = ["base_estimator"]
        default_parameters = type(self._clf)().get_params()
        for key, data in optimum_parameters.items():
            if (
                key not in default_parameters
                or default_parameters[key] != data
            ) and key not in excluded:
                result[key] = data
        return result


class ModelStree(ModelBase):
    def __init__(self, random_state: Optional[int] = None) -> None:
        self._clf = Stree()
        super().__init__(random_state)
        self._model_name = "stree"
        C = [0.05, 0.2, 0.55, 7, 55, 1e4]
        max_iter = [1e4, 1e5, 1e6]
        gamma = [1e-1, 1, 1e1]
        max_features = [None, "auto"]
        split_criteria = ["impurity", "max_samples"]
        self._linear = {
            "random_state": [self._random_state],
            "C": C,
            "max_iter": max_iter,
            "split_criteria": split_criteria,
            "max_features": max_features,
        }
        self._rbf = {
            "random_state": [self._random_state],
            "kernel": ["rbf"],
            "C": C,
            "gamma": gamma,
            "max_iter": max_iter,
            "split_criteria": split_criteria,
            "max_features": max_features,
        }
        self._poly = {
            "random_state": [self._random_state],
            "kernel": ["poly"],
            "degree": [3, 5],
            "C": C,
            "gamma": gamma,
            "max_iter": max_iter,
            "split_criteria": split_criteria,
            "max_features": max_features,
        }
        self._param_grid = [
            self._linear,
            self._poly,
            self._rbf,
        ]

    def select_params(self, kernel: str) -> None:
        if kernel == "linear":
            self._param_grid = [self._linear]
        elif kernel == "poly":
            self._param_grid = [self._poly]
        else:
            self._param_grid = [self._rbf]


class ModelSVC(ModelBase):
    def __init__(self, random_state: Optional[int] = None) -> None:
        super().__init__(random_state)
        self._clf = LinearSVC()
        self._model_name = "svc"
        max_iter = [1e4, 1e5, 1e6]
        self._param_grid = [
            {
                "random_state": [self._random_state],
                "C": [1, 55, 1e4],
                "max_iter": max_iter,
            },
        ]


class ModelDecisionTree(ModelBase):
    def __init__(self, random_state: Optional[int] = None) -> None:
        super().__init__(random_state)
        self._clf = DecisionTreeClassifier()
        self._model_name = "dtree"
        self._param_grid = [
            {
                "random_state": [self._random_state],
                "max_features": [None, "log2", "auto"],
            },
        ]


class Ensemble(ModelBase):
    def __init__(
        self,
        random_state: Optional[int] = 0,
        base_model: Union[BaseEnsemble, BaseEstimator] = None,
    ) -> None:
        super().__init__(random_state)
        self._base_model = base_model

    def merge_parameters(self, params: dict) -> dict:
        result = self._parameters.copy()
        for key, value in params.items():
            result[f"base_estimator__{key}"] = (
                value if isinstance(value, list) else [value]
            )
        return result

    def get_parameters(self) -> List[dict]:
        result = []
        for base_group in self._base_model.get_parameters():
            result.append(self.merge_parameters(base_group))
        return result


class ModelAdaBoost(Ensemble):
    def __init__(
        self, random_state: int, base_model: BaseEstimator = ModelStree
    ):
        # Build base_model
        super().__init__(
            random_state, base_model=base_model(random_state=random_state)
        )
        self._clf = AdaBoostClassifier(
            base_estimator=self._base_model.get_model(),
            random_state=random_state,
        )
        self._model_name = f"Adaboost_{self._base_model.__class__.__name__}"

    def get_parameters(self) -> List[dict]:
        self._parameters = {"n_estimators": [50, 100], "algorithm": ["SAMME"]}
        return super().get_parameters()


class ModelBagging(Ensemble):
    def __init__(
        self, random_state: int, base_model: BaseEstimator = ModelStree
    ) -> None:
        super().__init__(random_state, base_model=base_model(random_state))
        self._clf = BaggingClassifier(
            base_estimator=self._base_model.get_model(),
            random_state=random_state,
        )
        self._model_name = f"Bagging_{self._base_model.__class__.__name__}"

    def get_parameters(self) -> List[dict]:
        self._parameters = {
            "max_samples": [0.2, 0.4, 0.8, 1.0],
            "n_estimators": [50, 100],
            "max_features": [0.2, 0.6],
            "n_jobs": [-1],
        }
        return super().get_parameters()


class ModelOdte(Ensemble):
    def __init__(self, random_state: int, base_model=ModelStree) -> None:
        super().__init__(random_state, base_model=base_model(random_state))
        self._clf = Odte(
            base_estimator=Stree(random_state),
            random_state=random_state,
        )
        self._model_name = f"Odte_{self._base_model.__class__.__name__}"

    def get_parameters(self) -> List[dict]:
        self._parameters = {
            "max_samples": [0.2, 0.4, 0.8, 1.0],
            "n_estimators": [50, 100],
            "max_features": [0.2, 0.6, 1.0],
            "n_jobs": [-1],
        }
        return super().get_parameters()
