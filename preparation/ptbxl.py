"""
Based on code from `https://github.com/helme/ecg_ptbxl_benchmarking`
`master/code/experiments/scp_experiment.py`
"""
import os
import sys
import pickle

import numpy as np

sys.path.append("..")
import config
import utils

class DataPreparator():

    folds_type='strat'

    def __init__(
        self,
        task: str,
        min_samples: int,
        sampling_frequency: int,
        split_number: int=1
    ) -> None:
        """
        Args:
            task (str): Name of task ('all', 'diagnostic', 'subdiagnostic',
                                      'superdiagnostic', 'form', 'rhythm')
            min_samples (int):
            sampling_frequency (int): Sampling frequency (100 or 500).
            split_number (int): Select val and test fold index.
                val_fold_index (int): Index of stratifed split for validation dataset.
                test_fold_index (int): Index of stratifed split for test dataset.
                    (Other 8 indices not used will be treated as train_fold_indices)
        """

        assert(task in config.TASKS)
        self.task = task
        self.min_samples = min_samples
        self.sampling_frequency = sampling_frequency

        self.val_fold_index = config.split_settings[split_number]["val_index"]
        self.test_fold_index = config.split_settings[split_number]["test_index"]
        setting = f"{task}/val-{self.val_fold_index}_test-{self.test_fold_index}/"

        self.load_dir = os.path.join(config.root, config.dirname_ptbxl, "raw")
        self.save_dir = os.path.join(config.root, config.dirname_ptbxl, setting)
        os.makedirs(self.save_dir, exist_ok=True)

    def _split_data(self, data: np.ndarray, labels: np.ndarray, y_data: np.ndarray):
        """
        Args:
            data (np.ndarray): Array of shape (num_samples, sequence_length, 12).
            labels (np.ndarray): Array of shape (num_samples, ??)
            Y (np.ndarray): Array of shape (num_samples, ??)
        Returns:

        """
        test_target = labels.strat_fold == self.test_fold_index
        X_test = data[test_target]
        y_test = y_data[test_target]

        val_target = labels.strat_fold == self.val_fold_index
        X_val = data[val_target]
        y_val = y_data[val_target]

        train_target = ~val_target & ~test_target
        X_train = data[train_target]
        y_train = y_data[train_target]
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def _dump_data(self, X: np.ndarray, y: np.ndarray, datatype: str) -> None:
        """
        Args:
            X (np.ndarray):
            y (np.ndarray):
            datatype (str):
        Returns:
            None
        """
        print(f"Saving {datatype} data ...")
        X.dump(self.save_dir + f'X_{datatype}.npy', protocol=4)
        y.dump(self.save_dir + f'y_{datatype}.npy', protocol=4)

    def prepare(self):
        """
        Args:
        Returns:
        """
        # Load PTB-XL data
        data, raw_labels = utils.load_dataset(
            self.load_dir, self.sampling_frequency)

        # Preprocess label data
        labels = utils.compute_label_aggregations(
            raw_labels, self.load_dir, self.task)

        # Select relevant data and convert to one-hot
        data, labels, Y, _ = utils.select_data(
            data, labels, self.task, self.min_samples, self.save_dir)

        # Split data into train, valid, test
        (X_train, y_train), (X_val, y_val), (X_test, y_test) =\
            self._split_data(data, labels, Y)


        X_train, X_val, X_test = utils.preprocess_signals(
            X_train, X_val, X_test, self.save_dir)

        self._dump_data(X_train, y_train, "train")
        self._dump_data(X_val, y_val, "val")
        self._dump_data(X_test, y_test, "test")

if __name__ == "__main__":

    min_samples = 0
    sampling_frequency = 500
    split_number = int(sys.argv[1])
    for task in config.TASKS:
        print(f"Working on {task} data (split_number: {split_number})...")
        preparator = DataPreparator(task, min_samples, sampling_frequency,
                                    split_number)
        preparator.prepare()
