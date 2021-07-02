import sys
from datetime import datetime
from typing import Iterable
from importlib import import_module

import torch
import torch.nn as nn

sys.path.append("..")
import config
from codes.architectures.head_module import HeadModule, Classifier
from codes.data.dataloader import prepare_dataloader

import warnings
warnings.filterwarnings("ignore")

torch.manual_seed(0)
torch.backends.cudnn.deterministic=True

ParamsStringIgnore = ["device", "sequence_length",
                      "num_classes", "split_number"]

class BaseExecuter(object):

    def _get_timestamp(self) -> str:
        """
        Get timestamp in `yymmdd-hhmmss` format.
        Args:
            None
        Returns:
            timestamp (str): Time stamp in string.
        """
        timestamp = datetime.now()
        timestamp = timestamp.strftime('%Y%m%d-%H%M%S')[2:]
        return timestamp

    def _prepare_param_string(self) -> str:
        """
        Args:
            None
        Returns:
            param_string (str):
        """
        param_string = ""
        for key, value in self.args.__dict__.items():
            if key in ParamsStringIgnore:
                continue
            param_string += f"{key}-{value}_"
        param_string = param_string[:-1] # Remove last '_'
        return param_string

    def _prepare_model(self) -> nn.Module:
        """
        Load network architecture class from `codes.architectures`
        Args:
            None
        Returns:
            model (nn.Module):
        """
        params = {
            "backbone_out_dim": self.args.backbone_out_dim
        }
        if self.args.model.startswith("effnet"):
            params["sequence_length"] = self.args.sequence_length

        # Set output dim (if True -> multiclass case, else multilabel)
        if self.args.task.startswith("mc_"):
            out_dim = 3
        else:
            out_dim = config.num_classes_multilabel[self.args.task]

        # E.g
        # self.args.model = resnet1d-18"
        # then, modelfile = resnet1d
        # and, modelfile_path = `codes.architectures.resnet1d`
        modelfile = config.modelname_routing[self.args.model][0]
        modelfile_path = f"codes.architectures.{modelfile}"

        # Import model class function from modelfile_path file.
        model_module = import_module(modelfile_path)
        model_class = config.modelname_routing[self.args.model][1]

        BackboneModel = model_module.__dict__[model_class]
        backbone = BackboneModel(**params)
        # Used both for multi-label and multi-class classification
        heads = HeadModule(self.args.backbone_out_dim, out_dim)
        model = Classifier(backbone, heads)

        print(f"Loaded {model_class} ...")
        return model

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
        loader = prepare_dataloader(
            self.args.task, self.data_loc, datatype, self.args.bs,
            self.args.seed, self.args.freq, self.args.length, is_train)
        return loader

    def run(self):
        """
        Run training of model.

        Args:
            None
        Returns:
            None
        """
        raise NotImplementedError

if __name__ == "__main__":
    pass
