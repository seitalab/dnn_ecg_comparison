import os
import sys
from typing import Iterable

import torch
import numpy as np

sys.path.append("..")
import config
from codes.functions.train_model import ModelTrainerMC as Trainer
from codes.data.dataloader import prepare_dataloader_multiclass
from execute_base import BaseExecuter

import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.deterministic=True

class TrainExecuterMC(BaseExecuter):
    """Execute train code for multiclass classification"""

    def __init__(self, args, dataset) -> None:
        assert(dataset in config.DATASETS)

        self.args = args
        self.dataset = dataset
        if dataset == "ptbxl":
            self.data_loc = os.path.join(
                config.root, config.dirname_ptbxl, "all")
        elif dataset == "g12ec":
            self.data_loc = os.path.join(
                config.root, config.dirname_g12ec, "processed")
        elif dataset == "cpsc":
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

    def _prepare_dataloader(self, datatype: str,
                            is_train: bool=False) -> Iterable:
        """
        Prepare dataloader for training and validating model.
        Args:
            datatype (str): Type of dataset ("train", "val", "test").
            is_train (bool): Dataloader as train mode or not.
        Returns:
            loader (Iterable): Dataloader.
        """
        print("Preparing {} dataloader ...".format(datatype))
        loader = prepare_dataloader_multiclass(
            self.args.task, self.dataset, self.data_loc, datatype, self.args.bs,
            self.args.seed, self.args.freq, self.args.length, is_train)
        return loader

    def _calc_class_weight(self, labels: np.ndarray) -> np.ndarray:
        """
        Calculate class weight for target dx and others (1 for normal labels).

        Args:
            labels (np.ndarray): Label data array of shape [num_sample, num_classes]
        Returns:
            class_weight (np.ndarray):
        """
        num_samples = labels.shape[0]

        # Index for normal and target dx in target dataset.
        target_dx = self.args.task[3:] # eg. `mc_AF`
        normal_index = config.MULTICLASS_LABELS_INDEX["Normal"][self.dataset]
        target_index = config.MULTICLASS_LABELS_INDEX[target_dx][self.dataset]

        # Extract normal and target dx labels
        normal_labels = labels[:, normal_index]
        target_labels = labels[:, target_index]
        # Validate normal and target dx label do not overlap
        # assert((normal_labels & target_labels).sum() == 0)

        num_normal = normal_labels.sum()
        num_target = target_labels.sum()
        num_others = num_samples - (num_normal + num_target)

        class_weights = [1, num_target/num_normal, num_others/num_normal]
        return np.array(class_weights)

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
        class_weight = self._calc_class_weight(train_loader.dataset.label)

        self.trainer.set_model(model)
        self.trainer.set_optimizer(self.args.lr)
        self.trainer.set_lossfunc(class_weight)

        self.trainer.save_params(self.args)
        self.trainer.run(train_loader, valid_loader)

if __name__ == "__main__":
    from hyperparams import args

    executer = TrainExecuterMC(args, "cpsc")
    executer.run()
