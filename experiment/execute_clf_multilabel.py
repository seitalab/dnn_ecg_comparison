import os
import sys
from typing import List, Tuple
from argparse import Namespace

import numpy as np
import pandas as pd

sys.path.append("..")
import config
from execute_eval import EvalExecuter
from execute_train import TrainExecuter

colnames = ["task", "seed", "Mean score", "Bootstrap mean",
            "Bootstrap 5%", "Bootstrap 95%", "save_loc"]

class MultiLabelClfExecuter(object):
    """
    Execute multilabel classification tasks with different split.
    """

    def __init__(self, modelname: str, device: str) -> None:

        self.freq = config.freq
        self.length = config.length
        self.optim = config.optimizer
        self.epoch = config.num_epochs
        self.eval_every = config.eval_every
        self.patience = config.patience
        self.backbone_out_dim = config.backbone_out_dim

        self.device = device
        self.modelname = modelname

    def _prepare_params(
        self,
        task: str,
        seed: int,
        batch_size: int,
        learning_rate: float
    ) -> Namespace:
        """
        Args:
            task (str): task
            seed (int): seed
            batch_size (int): batch size
            learning_rate (float): learning rate
        Returns:
            params (Namespace):
        """
        params = Namespace(
            task = task,
            seed = seed,
            ep = self.epoch,
            freq = self.freq,
            length = self.length,

            model = self.modelname,
            optim = self.optim,
            backbone_out_dim = self.backbone_out_dim,
            device = self.device,
            eval_every = self.eval_every,
            patience=self.patience,

            bs = batch_size,
            lr = learning_rate,

            split_number = seed,
            num_classes = config.num_classes_multilabel[task],
            sequence_length = self.freq * self.length
        )
        return params

    def _load_gridsearch_result(self) -> pd.DataFrame:
        """
        Args:
            None
        Returns:
            df_result (pd.DataFrame):
        """
        target_file = config.gridsearch_result_loc + f"/{self.modelname}.csv"
        gridsearch_result = pd.read_csv(target_file)
        return gridsearch_result

    def _select_hyperparams(self) -> Tuple[int, float]:
        """
        Args:
            None
        Returns:
            batch_size (int):
            learning_rate (float):
        """
        gridsearch_result = self._load_gridsearch_result()
        scores = gridsearch_result.loc[:, config.selection_criteria].values
        best_score_idx = np.argmax(scores)
        batch_size = gridsearch_result.loc[best_score_idx, "batch_size"]
        learning_rate = gridsearch_result.loc[best_score_idx, "learning_rate"]
        return int(batch_size), float(learning_rate)

    def _prepare_result_df(
        self,
        task: str,
        seed: int,
        result: List,
        saved_loc: str
        ) -> pd.DataFrame:
        """
        Args:
            task (str):
            seed (int):
            result (List): List of test set results.
            save_loc (str):
        Returns:
            df_result (pd.DataFrame):
        """
        result_list = [
            task, seed, round(result[1], 4),  round(result[2][0], 4),
            round(result[2][1], 4),  round(result[2][2], 4), saved_loc
        ]
        df_result = pd.DataFrame([result_list], columns=colnames)
        return df_result

    def run(self):
        """
        Args:

        Returns:

        """

        df_result = pd.DataFrame(columns=colnames)
        batch_size, learning_rate = self._select_hyperparams()

        os.makedirs(config.multilabel_result_loc, exist_ok=True)
        savename = config.multilabel_result_loc + f"/{self.modelname}.csv"

        for task in config.MULTILABEL_TASKS:
            for seed in config.SEEDS:
                params = self._prepare_params(
                    task, seed, batch_size, learning_rate)
                train_executer = TrainExecuter(params)
                train_executer.run()
                saved_loc = train_executer.save_dir

                eval_executer = EvalExecuter(saved_loc, self.device)
                test_result = eval_executer.run()

                _result = self._prepare_result_df(
                    task, seed, test_result, saved_loc)
                df_result = df_result.append(_result, ignore_index=True)
                # save temporal result
                df_result.to_csv(savename.replace(".csv", "_tmp.csv"))

        # Overwrite result
        df_result.to_csv(savename)
        print(f"Saved at {savename}")

if __name__ == "__main__":
    # modelname = "resnet1d-18"
    # device = "cuda:0"
    import sys
    modelname = sys.argv[1]
    device = sys.argv[2]

    mce = MultiLabelClfExecuter(modelname, device)
    mce.run()
