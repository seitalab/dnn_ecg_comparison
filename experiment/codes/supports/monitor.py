import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

class Monitor(object):

    def __init__(self):
        self.num_data = 0
        self.total_loss = 0
        self.ytrue_record = None
        self.ypred_record = None

    def _concat_array(self, record, new_data: np.array):
        """
        Args:

        Returns:

        """
        if record is None:
            return new_data
        else:
            return np.concatenate([record, new_data])

    def store_loss(self, loss: float, num_data: int) -> None:
        """
        Args:
            loss (float): Mini batch loss value.
            num_data (int): Number of data in mini batch.
        Returns:
            None
        """
        self.total_loss += loss
        self.num_data += num_data

    def store_result(self, y_trues: np.ndarray, y_preds: np.ndarray) -> None:
        """
        Args:
            y_trues (np.ndarray):
            y_preds (np.ndarray): Array with 0 - 1 values.
        Returns:
            None
        """
        y_trues = y_trues.cpu().detach().numpy()
        y_preds = y_preds.cpu().detach().numpy()

        self.ytrue_record = self._concat_array(self.ytrue_record, y_trues)
        self.ypred_record = self._concat_array(self.ypred_record, y_preds)
        assert(len(self.ytrue_record) == len(self.ypred_record))

    def average_loss(self) -> float:
        """
        Args:
            None
        Returns:
            average_loss (float):
        """
        return self.total_loss / self.num_data

    def _find_optimal_threshold(self, ytrue, ypred):
        """
        Find optimal cutoff threshold for given class prediction and true labels.
        From `https://github.com/helme/ecg_ptbxl_benchmarking/blob/06187fbc28992f26e15e44058d49f92e1485b079/code/utils/utils.py#L78`.

        Args:
            ytrue (np.ndarray):
            ypred (np.ndarray):
        Returns:

        """
        fpr, tpr, threshold = roc_curve(ytrue, ypred)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = threshold[optimal_idx]
        return optimal_threshold

    def fmax_score(self):
        """
        ```
        the threshold is optimized on the respective test set for
        each classification task and classifier under consideration.
        ```
        Details of metric from CAFA challenge paper
        (`https://genomebiology.biomedcentral.com/track/pdf/10.1186/s13059-016-1037-6.pdf`)

        Args:
            None
        Returns:

        """
        num_classes = self.ytrue_record.shape[1]
        fmax_scores = []

        # Get optimal threshold for each classes and calculate Fmax score.
        for i in range(num_classes):
            threshold = self._find_optimal_threshold(
                self.ytrue_record[:, i], self.ypred_record[:, i])
            ypred = self.ypred_record[:, i] > threshold
            fmax_scores.append(f1_score(self.ytrue_record[:, i], ypred))
        return fmax_scores

    def macro_auc_roc(self):
        """
        Args:

        Returns:

        """
        score = roc_auc_score(
            self.ytrue_record, self.ypred_record, average='macro')
        return score

    def macro_f1(self):
        """
        Args:

        Returns:

        """
        y_preds = np.argmax(self.ypred_record, axis=1)
        score = f1_score(self.ytrue_record, y_preds, average='macro')
        return score
