import torch
import torch.nn.functional as F
import numpy as np

def get_appropriate_bootstrap_samples(y_true, n_bootstraping_samples):
    """
    From `https://github.com/helme/ecg_ptbxl_benchmarking/blob/06187fbc28992f26e15e44058d49f92e1485b079/code/utils/utils.py#L68`
    """
    samples=[]
    while True:
        ridxs = np.random.randint(0, len(y_true), len(y_true))
        if y_true[ridxs].sum(axis=0).min() != 0:
            samples.append(ridxs)
            if len(samples) == n_bootstraping_samples:
                break
    return samples


def aggregator(model, X):
    """
    ```
    These predictions are then aggregated using the element-wise maximum
    (or mean in case of age and gender prediction)
    ```
    From paper's appendix I => We use maximum.

    Args:
        model (nn.Module):
        X (torch.Tensor): Tensor of shape (batch_size, 12, num_split, sequence_length).
    Returns:
        aggregated_preds (torch.Tensor): Tensor of shape (batch_size, num_classes).
    """
    X = torch.transpose(X, 1, 2)
    aggregated_preds = []
    for i in range(X.size(0)):
        y_preds = model(X[i]) # X[i]: (num_split, 12, sequence_length)
        _aggregated_preds, _ = torch.max(y_preds, axis=0)
        aggregated_preds.append(_aggregated_preds)
    aggregated_preds = torch.stack(aggregated_preds)
    return aggregated_preds

class EarlyStopper(object):

    def __init__(self, mode: str, patience: int):
        """
        Args:
            mode (str): max or min
            patience (int):
        Returns:
            None
        """
        assert(mode in ["max", "min"])
        self.mode = mode

        self.patience = patience
        self.num_bad_count = 0

        if mode == "max":
            self.best = -1 * np.inf
        else:
            self.best = np.inf

    def stop_training(self, metric: float):
        """
        Args:
            metric (float):
        Returns:
            stop_train (bool):
        """
        if self.mode == "max":

            if metric <= self.best:
                self.num_bad_count += 1
            else:
                self.num_bad_count = 0
                self.best = metric

        else:

            if metric >= self.best:
                self.num_bad_count += 1
            else:
                self.num_bad_count = 0
                self.best = metric

        if self.num_bad_count > self.patience:
            stop_train = True
        else:
            stop_train = False
        return stop_train
