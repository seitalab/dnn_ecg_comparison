import os
import sys
from typing import List
from argparse import Namespace

import pandas as pd

sys.path.append("..")
import config
from execute_eval import EvalExecuter
from execute_train import TrainExecuter

colnames = ["learning_rate", "batch_size",
            "Mean score", "Bootstrap mean",
            "Bootstrap 5%", "Bootstrap 95%", "save_loc"]

class GridSearchExecuter:

    def __init__(self, modelname: str, device: str) -> None:

        self.task = "all" # Use `all` to select hyper parameters
        self.seed = 1 # Evaluate with seed value of 1

        self.freq = config.freq
        self.length = config.length
        self.optim = config.optimizer
        self.epoch = config.num_epochs
        self.eval_every = config.eval_every
        self.patience = config.patience
        self.backbone_out_dim = config.backbone_out_dim

        self.num_classes = config.num_classes_multilabel[self.task]
        self.device = device
        self.modelname = modelname

    def _prepare_params(self, batch_size: int, learning_rate: float
        ) -> Namespace:
        """
        Args:
            batch_size (int): batch size
            learning_rate (float): learning rate
        Returns:
            params (Namespace):
        """
        params = Namespace(
            task = self.task,
            seed = self.seed,
            ep = self.epoch,
            freq = self.freq,
            length = self.length,

            model = self.modelname,
            optim = self.optim,
            backbone_out_dim = self.backbone_out_dim,
            device = self.device,
            eval_every = self.eval_every,
            patience = self.patience,

            bs = batch_size,
            lr = learning_rate,

            split_number = self.seed,
            num_classes = self.num_classes,
            sequence_length = self.freq * self.length
        )
        return params

    def _prepare_result_df(
        self, lr: float, bs: int, result: List, saved_loc: str
        ) -> pd.DataFrame:
        """
        Args:
            lr (float):
            bs (int):
            result (List): List of test set results.
            save_loc (str):
        Returns:
            df_result (pd.DataFrame):
        """
        result_list = [
            lr, bs, round(result[1], 4),  round(result[2][0], 4),
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

        os.makedirs(config.gridsearch_result_loc, exist_ok=True)
        savename = config.gridsearch_result_loc + f"/{self.modelname}.csv"

        for lr in config.lr_range:
            for bs in config.bs_range:
                params = self._prepare_params(bs, lr)
                train_executer = TrainExecuter(params)
                train_executer.run()
                saved_loc = train_executer.save_dir

                eval_executer = EvalExecuter(saved_loc, self.device)
                test_result = eval_executer.run()

                _result = self._prepare_result_df(
                    lr, bs, test_result, saved_loc)
                df_result = df_result.append(_result, ignore_index=True)
                df_result.to_csv(savename.replace(".csv", "_tmp.csv"))

        df_result.to_csv(savename)
        print(f"Saved at {savename}")

if __name__ == "__main__":
    modelname = sys.argv[1]
    device = sys.argv[2]

    gs = GridSearchExecuter(modelname, device)
    gs.run()
