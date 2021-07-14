import os
import sys

import torch
import numpy as np

sys.path.append("..")
import config
from codes.functions.train_model import ModelTrainer as Trainer
from execute_base import BaseExecuter

import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.deterministic=True

class TrainExecuter(BaseExecuter):

    def __init__(self, args) -> None:

        self.args = args
        if args.task in config.TASKS:
            self.data_loc = os.path.join(config.root, config.dirname_ptbxl, args.task)
        elif args.task == "g12ec":
            self.data_loc = os.path.join(
                config.root, config.dirname_g12ec, "processed")
        elif args.task == "cpsc":
            self.data_loc = os.path.join(
                config.root, config.dirname_cpsc, "processed_modelclf")
        else:
            raise NotImplementedError

        timestamp = self._get_timestamp()
        param_string = self._prepare_param_string()

        save_dir = os.path.join(config.save_dir, "model", param_string, timestamp)
        log_dir = os.path.join(config.save_dir, "logs", param_string, timestamp)

        self.save_dir = save_dir # Used when gridsearching

        self.trainer = Trainer(
            args.ep, save_dir=save_dir, log_dir=log_dir, patience=args.patience,
            eval_every=args.eval_every, device=args.device)

    def run(self):
        """
        Run training of model.
        Args:
            None
        Returns:
            None
        """

        model = self._prepare_model()
        train_loader = self._prepare_dataloader("train", is_train=True)
        valid_loader = self._prepare_dataloader("val")

        self.trainer.set_model(model)
        self.trainer.set_optimizer(self.args.lr)
        self.trainer.set_lossfunc()

        self.trainer.save_params(self.args)
        self.trainer.run(train_loader, valid_loader)

if __name__ == "__main__":
    from hyperparams import args

    executer = TrainExecuter(args)
    executer.run()
