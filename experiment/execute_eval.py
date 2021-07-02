import os
import sys
import pickle
from typing import Iterable, List

import torch

sys.path.append("..")
import config
from execute_train import TrainExecuter
from codes.functions.evaluate_model import ModelEvaluator as Evaluator

import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.deterministic=True

class EvalExecuter(TrainExecuter):

    def __init__(self, eval_target: str, device: str="cpu"):

        self.param_file = os.path.join(eval_target, "params.pkl")
        self.weightfile = os.path.join(eval_target, "net.pth")

        self.args = self._load_params()
        self.evaluator = Evaluator(device)

        if self.args.task in config.TASKS:
            self.data_loc = os.path.join(config.root, config.dirname, self.args.task)
        elif self.args.task == "g12ec":
            self.data_loc = os.path.join(
                config.root, config.dirname_g12ec, "processed")
        elif self.args.task == "cpsc":
            self.data_loc = os.path.join(
                config.root, config.dirname_cpsc, "processed_modelclf")
        else:
            raise NotImplementedError

    def _load_params(self):
        """
        Load pickled params.
        Args:
            None
        Returns:
            params
        """
        with open(self.param_file, "rb") as fp:
            params = pickle.load(fp)
        return params

    def run(self) -> List:
        """
        Run evaluation of model.
        Args:
            None
        Returns:
            test_result (list): List of result values (loss is not used, always 0).
                                [loss, score, [Bootstrap mean, Bootstrap 5%, Bootstrap 95%]].
        """

        model = self._prepare_model()

        self.evaluator.set_model(model)
        self.evaluator.set_weight(self.weightfile)
        self.evaluator.set_optimizer(self.args.lr)
        self.evaluator.set_lossfunc()

        # print("-"*80)
        # print("Working on train set ...")
        # train_loader = self._prepare_dataloader("train", is_train=False)
        # train_result = self.evaluator.run(train_loader)
        # print(f'Score: {train_result[1]:.4f}')

        print("-"*80)
        print("Working on valid set ...")
        valid_loader = self._prepare_dataloader("val", is_train=False)
        valid_result = self.evaluator.run(valid_loader)
        print(f'Score: {valid_result[1]:.4f}')

        print("-"*80)
        print("Working on test set ...")
        test_loader = self._prepare_dataloader("test", is_train=False)
        test_result = self.evaluator.run(test_loader, apply_bootstrap=True)
        print(f'Score: {test_result[1]:.4f}')
        print(f'Bootstrap mean: {test_result[2][0]:.4f}')
        print(f'\t(5%: {test_result[2][1]:.4f}, 95%: {test_result[2][2]:.4f})')
        print("-"*80)

        return test_result

if __name__ == "__main__":
    import sys
    from result_dict import models_dict

    eval_target = models_dict[sys.argv[1]]
    device = sys.argv[2]

    executer = EvalExecuter(eval_target, device)
    executer.run()
