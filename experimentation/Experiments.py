import json
import os
import time
import warnings

from sklearn.model_selection import GridSearchCV, cross_validate

from . import Models
from .Database import Hyperparameters, Outcomes
from .Sets import Datasets


class Experiment:
    def __init__(
        self,
        random_state: int,
        model: str,
        host: str,
        set_of_files: str,
        kernel: str,
    ) -> None:
        self._random_state = random_state
        self._model_name = model
        self._set_of_files = set_of_files
        self._type = getattr(
            Models,
            f"Model{model[0].upper() + model[1:]}",
        )
        self._clf = self._type(random_state=self._random_state)
        self._host = host
        # used in gridsearch with ensembles to take best hyperparams of
        # base class or gridsearch these hyperparams as well
        self._base_params = "any"
        self._kernel = kernel

    def set_base_params(self, base_params: str) -> None:
        self._base_params = base_params

    def cross_validation(self, dataset: str) -> None:
        hyperparams = Hyperparameters(host=self._host, model=self._model_name)
        try:
            parameters, normalize, standardize = hyperparams.get_params(
                dataset
            )
        except ValueError:
            print(f"*** {dataset} not trained")
            return
        datasets = Datasets(
            normalize=normalize,
            standardize=standardize,
            set_of_files=self._set_of_files,
        )
        parameters = json.loads(parameters)
        X, y = datasets.load(dataset)
        # init cross validation object just in case consecutive experiments
        self._clf = self._type(random_state=self._random_state)
        model = self._clf.get_model().set_params(**parameters)
        self._num_warnings = 0
        warnings.warn = self._warn
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # Also affect subprocesses
            os.environ["PYTHONWARNINGS"] = "ignore"
            results = cross_validate(
                model, X, y, return_train_score=True, n_jobs=-1
            )
        outcomes = Outcomes(host=self._host, model=self._model_name)
        parameters = json.dumps(parameters, sort_keys=True)
        outcomes.store(dataset, normalize, standardize, parameters, results)
        if self._num_warnings > 0:
            print(f"{self._num_warnings} warnings have happend")

    def grid_search(
        self, dataset: str, normalize: bool, standardize: bool
    ) -> None:
        """First of all if the modle is an ensemble search for the best
        hyperparams found in gridsearch for base model and overrides
        normalize and standardize
        """
        hyperparams = Hyperparameters(host=self._host, model=self._model_name)
        model = self._clf.get_model()
        if self._kernel != "any":
            # set parameters grid to only one kernel
            if isinstance(self._clf, Models.Ensemble):
                self._clf._base_model.select_params(self._kernel)
            else:
                self._clf.select_params(self._kernel)
        hyperparameters = self._clf.get_parameters()
        grid_type = "gridsearch"
        if (
            isinstance(self._clf, Models.Ensemble)
            and self._base_params == "best"
        ):
            hyperparams_base = Hyperparameters(
                host=self._host, model=self._clf._base_model.get_model_name()
            )
            try:
                # Get best hyperparameters obtained in gridsearch for base clf
                (
                    base_hyperparams,
                    normalize,
                    standardize,
                ) = hyperparams_base.get_params(dataset)
                # Merge hyperparameters with the ensemble ones
                base_hyperparams = json.loads(base_hyperparams)
                hyperparameters = self._clf.merge_parameters(base_hyperparams)
                grid_type = "gridbest"
            except ValueError:
                pass
        dt = Datasets(
            normalize=normalize,
            standardize=standardize,
            set_of_files=self._set_of_files,
        )
        X, y = dt.load(dataset)
        self._num_warnings = 0
        warnings.warn = self._warn
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # Also affect subprocesses
            os.environ["PYTHONWARNINGS"] = "ignore"
            grid_search = GridSearchCV(
                model,
                return_train_score=True,
                param_grid=hyperparameters,
                n_jobs=-1,
                verbose=1,
            )
            start_time = time.time()
            grid_search.fit(X, y)
            time_spent = time.time() - start_time
        parameters = json.dumps(
            self._clf.modified_parameters(
                grid_search.best_estimator_.get_params()
            ),
            sort_keys=True,
        )
        hyperparams.store(
            dataset,
            time_spent,
            grid_search,
            parameters,
            normalize,
            standardize,
            grid_type,
        )
        if self._num_warnings > 0:
            print(f"{self._num_warnings} warnings have happend")

    def _warn(self, *args, **kwargs) -> None:
        self._num_warnings += 1
