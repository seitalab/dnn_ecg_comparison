from typing import Iterable, Tuple

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score

from codes.functions.train_model import ModelTrainer
from codes.supports import utils
from codes.supports.monitor import Monitor

class ModelEvaluator(ModelTrainer):

    def __init__(self, device: str="cpu"):

        self.device = device
        self.model = None

    def set_weight(self, weight_file: str) -> None:
        """
        Set trained weight to model.
        Args:
            weight_file (str):
        Returns:
            None
        """
        assert(self.model is not None)

        self.model.to("cpu")
        self.model.load_state_dict(torch.load(weight_file, map_location="cpu"))
        self.model.to(self.device)

    def _evaluate(self, iterator: Iterable
        ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Run evaluation mode iteration with boot strapping.

        Args:
            iterator (Iterable):
        Returns:
            loss (float):
            score (float):
            ytrues (np.ndarray):
            ypreds (np.ndarray):
        """

        monitor = Monitor()
        self.model.eval()

        with torch.no_grad():
            for X, y in tqdm(iterator):
                X = X.float().to(self.device)
                y = y.float().to(self.device)

                y_pred = utils.aggregator(self.model, X)

                monitor.store_loss(0, len(X))
                monitor.store_result(y, y_pred)

        loss = monitor.average_loss()
        score = monitor.macro_auc_roc()
        ytrues = monitor.ytrue_record
        ypreds = monitor.ypred_record
        return loss, score, ytrues, ypreds

    def _evaluate_bootstrap(self, ytrues: np.ndarray, ypreds: np.ndarray
        ) -> np.ndarray:
        """
        Args:
            ytrues (np.ndarray):
            ypreds (np.ndarray):
        Returns:
            scores (np.ndarray):
        """
        print("Bootstrapping test set result ...")
        n_bootstrap = 100
        samples = utils.get_appropriate_bootstrap_samples(ytrues, n_bootstrap)
        scores = []
        for sample in tqdm(samples):
            score = roc_auc_score(ytrues[sample], ypreds[sample], average="macro")
            scores.append(score)
        scores = np.array(scores)
        return scores

    def run(self, loader: Iterable, apply_bootstrap: bool=False) -> None:
        """
        Args:
            loader (DataLoader): Dataloader for validation data.
        Returns:
            loss (float):
            score (float):
        """

        loss, score, ytrues, ypreds = self._evaluate(loader)
        if not apply_bootstrap:
            return loss, score
        else:
            bootstrap_scores = self._evaluate_bootstrap(ytrues, ypreds)
            b_mean = bootstrap_scores.mean()
            b_floor = np.quantile(bootstrap_scores, 0.05)
            b_ceil = np.quantile(bootstrap_scores, 0.95)
            return loss, score, (b_mean, b_floor, b_ceil)

class ModelEvaluatorMC(ModelEvaluator):

    def _evaluate(self, iterator: Iterable
        ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Run evaluation mode iteration with boot strapping.

        Args:
            iterator (Iterable):
        Returns:
            loss (float):
            score (float):
            ytrues (np.ndarray):
            ypreds (np.ndarray):
        """

        monitor = Monitor()
        self.model.eval()

        with torch.no_grad():
            for X, y in tqdm(iterator):
                X = X.float().to(self.device)
                y = y.long().to(self.device)

                y_pred = utils.aggregator(self.model, X)

                monitor.store_loss(0, len(X))
                monitor.store_result(y, y_pred)

        loss = monitor.average_loss()
        score = monitor.macro_f1()
        ytrues = monitor.ytrue_record
        ypreds = monitor.ypred_record
        return loss, score, ytrues, ypreds

    def _evaluate_bootstrap(self, ytrues: np.ndarray, ypreds: np.ndarray
        ) -> np.ndarray:
        """
        Args:
            ytrues (np.ndarray):
            ypreds (np.ndarray):
        Returns:
            scores (np.ndarray):
        """
        print("Bootstrapping test set result ...")
        n_bootstrap = 100
        samples = utils.get_appropriate_bootstrap_samples(ytrues, n_bootstrap)
        scores = []
        for sample in tqdm(samples):
            _ypreds = np.argmax(ypreds[sample], axis=1)
            score = f1_score(ytrues[sample], _ypreds, average="macro")
            scores.append(score)
        scores = np.array(scores)
        return scores
