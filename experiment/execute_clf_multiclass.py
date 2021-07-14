import os
import sys
from typing import List, Tuple
from argparse import Namespace

import numpy as np
import pandas as pd

sys.path.append("..")
import config
from execute_eval_multiclass import EvalExecuterMC
from execute_train_multiclass import TrainExecuterMC
from execute_clf_multilabel import MultiLabelClfExecuter

colnames = ["task", "dataset", "seed", "Mean score", "Bootstrap mean",
            "Bootstrap 5%", "Bootstrap 95%", "save_loc"]

class MultiClassClfExecuter(MultiLabelClfExecuter):
    """
    Execute multiclass classification tasks with different split.
    """

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
            num_classes = 3, # Fix to 3 (normal, target_dx, other).
            sequence_length = self.freq * self.length
        )
        return params

    def _prepare_result_df(
        self,
        task: str,
        dataset: str,
        seed: int,
        result: List,
        saved_loc: str
        ) -> pd.DataFrame:
        """
        Args:
            task (str):
            dataset (str): Name of dataset.
            seed (int):
            result (List): List of test set results.
            save_loc (str):
        Returns:
            df_result (pd.DataFrame):
        """
        result_list = [
            task, dataset, seed, round(result[1], 4), round(result[2][0], 4),
            round(result[2][1], 4), round(result[2][2], 4), saved_loc
        ]
        df_result = pd.DataFrame([result_list], columns=colnames)
        return df_result

    def _is_target_task(self, dataset: str, task: str) -> bool:
        """
        Args:
            dataset (str): Dataset name.
            task (str): Name of task (`mc_<TARGET_DX>`, eg. `mc_AF`)
        Returns:
            is_target (bool): True if target index is not None.
        """
        # Validate normal class index exist
        assert(config.MULTICLASS_LABELS_INDEX["Normal"][dataset] is not None)

        assert(task.startswith("mc_"))
        target_dx = task[3:]
        target_index = config.MULTICLASS_LABELS_INDEX[target_dx][dataset]
        # False: if (STE, ptbxl), (STD, g12ec)
        return target_index is not None

    def run(self):
        """
        Args:

        Returns:

        """

        df_result = pd.DataFrame(columns=colnames)
        batch_size, learning_rate = self._select_hyperparams()

        os.makedirs(config.multiclass_result_loc, exist_ok=True)
        savename = config.multiclass_result_loc + f"/{self.modelname}.csv"

        for task in config.MULTICLASS_TASK:
            for dataset in config.DATASETS:
                for seed in config.SEEDS:
                    # Skip if target dx does not exist in dataset.
                    if not self._is_target_task(dataset, task):
                        continue

                    params = self._prepare_params(
                        task, seed, batch_size, learning_rate)
                    train_executer = TrainExecuterMC(params, dataset)
                    train_executer.run()
                    saved_loc = train_executer.save_dir

                    eval_executer = EvalExecuterMC(
                        saved_loc, dataset, self.device)
                    test_result = eval_executer.run()

                    _result = self._prepare_result_df(
                        task, dataset, seed, test_result, saved_loc)
                    df_result = df_result.append(_result, ignore_index=True)
                    df_result.to_csv(savename.replace(".csv", "_tmp.csv"))

        # Save final result
        df_result.to_csv(savename)
        print(f"Saved at {savename}")

if __name__ == "__main__":
    # modelname = "resnet1d-18"
    # device = "cuda:0"
    import sys
    modelname = sys.argv[1]
    device = sys.argv[2]

    mce = MultiClassClfExecuter(modelname, device)
    mce.run()
