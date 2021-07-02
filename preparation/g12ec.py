import os
import sys
from collections import Counter
from typing import List, Tuple, Type
from glob import glob

import wfdb
import numpy as np
from tqdm import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

sys.path.append("..")
import config
import utils

class G12ECPreparator:

    def __init__(self,
        sampling_frequency: int=500,
        split_number: int=1,
        min_sample_ratio: float=0.01,
    ) -> None:
        """
        Args:
            sampling_frequency (int): Sampling frequency (500).
            split_number (int): Seed　value for train test split.
            min_samples (float): Number of minimum sample ratio.
        """

        self.sampling_frequency = sampling_frequency
        self.split_number = split_number
        self.min_sample_ratio = min_sample_ratio

        self.load_dir = os.path.join(config.data_root, "WFDB")
        self.save_dir = os.path.join(config.data_root, "processed")
        os.makedirs(self.save_dir, exist_ok=True)

    def _open_heafile(self, hea_file: str) -> Type[wfdb.io.record.Record]:
        """
        Args:
            hea_file (str): Path to hea file.
        Returns:
            waveform_data ():
        """
        basename, _ = os.path.splitext(hea_file)
        waveform_data = wfdb.rdrecord(basename)
        return waveform_data

    def _load_data(self) -> Tuple[np.ndarray, List, Tuple[List, List]]:
        """
        Args:
            None
        Returns:
            signals (np.ndarray): Array of 12lead ECG signals with length num_samples.
                            (Each elements are array of shape [sequence_length, 12])
            dxs (List): List of diagnosis ids
            demographics (Tuple): Tuple of list of sex and age of each data.
        """
        hea_files = sorted(glob(self.load_dir + "/*.hea"))
        signals = []
        dxs, sexs, ages = [], [], []
        for hea_file in tqdm(hea_files):
            data = self._open_heafile(hea_file)
            assert(data.n_sig == 12)
            assert(data.fs == 500)
            assert(data.sig_name == config.g12ec_lead_order)
            signal = np.nan_to_num(data.p_signal, 0)
            signals.append(signal)

            sexs.append(data.comments[1])
            ages.append(data.comments[0])
            dxs.append(data.comments[2])
        return np.array(signals, dtype=object), dxs, (sexs, ages)

    def _align_signal_length(self, signals: np.ndarray):
        """
        Args:
            signals (np.ndarray):
        Returns:
            aligned_signals (np.ndarray):
        """
        aligned_signals = []
        for signal in signals:
            # Padding
            signal_length = signal.shape[0]
            if signal_length > config.g12ec_default_signal_length:
                raise ValueError(f"Signal length {signal_length} exceeded default_signal_length.")
            elif signal_length < config.g12ec_default_signal_length:
                pad_length = config.g12ec_default_signal_length - signal_length
                pad = np.zeros([pad_length, signal.shape[1]])
                signal = np.concatenate([pad, signal], axis=0)
            aligned_signals.append(signal)
        aligned_signals = np.stack(aligned_signals)
        return aligned_signals

    def _preprocess_signal(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Scale data.

        Args:
            X_train (np.ndarray): Array of arrays of shape [(sequence_length, 12), (..), .., (..)].
            X_val (np.ndarray): Array of arrays of shape [(sequence_length, 12), (..), .., (..)].
            X_test (np.ndarray): Array of arrays of shape [(sequence_length, 12), (..), .., (..)].
        Returns:
            X_train (np.ndarray): Array of arrays of shape [(sequence_length, 12), (..), .., (..)].
            X_val (np.ndarray): Array of arrays of shape [(sequence_length, 12), (..), .., (..)].
            X_test (np.ndarray): Array of arrays of shape [(sequence_length, 12), (..), .., (..)].
        """
        # apply padding
        X_train = self._align_signal_length(X_train)
        X_val = self._align_signal_length(X_val)
        X_test = self._align_signal_length(X_test)

        # apply scaling
        X_train, X_val, X_test = utils.preprocess_signals(
            X_train, X_val, X_test, self.save_dir, seed=self.split_number)

        return X_train, X_val, X_test

    def _process_label(self, labels: List) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            labels (List):
        Returns:
            processed_labels (np.ndarray): Array of shape (num_samples, num_labels).
            target_labels (np.ndarray): List of dx code corresponding to `processed_labels` index.
        """
        label_index = []
        for label in labels:
            # label = "Dx: XXXXX,YYYYY"
            label = label.replace("Dx: ", "")
            label = label.split(",")
            label_index += label

        # Select labels with more than `self.min_sample_ratio * len(labels)`.
        target_labels = []
        for idx, count in Counter(label_index).items():
            if count > int(self.min_sample_ratio * len(labels)):
                target_labels.append(idx)

        processed_labels = np.zeros([len(labels), len(target_labels)])
        for i, label in enumerate(labels):
            # label = "Dx: XXXXX,YYYYY"
            label = label.replace("Dx: ", "")
            label = label.split(",")
            for l in label:
                if l in target_labels:
                    processed_labels[i, target_labels.index(l)] = 1
        return processed_labels, np.array(target_labels)

    def _process_demographics(self, demographics: Tuple):
        """
        Args:
            demographics (Tuple[np.ndarray]):
        Returns:
            processed_demos (np.ndarray):
        """
        processed_demos = []
        sexs, ages = demographics
        for (sex, age) in zip(sexs, ages):
            sex = sex.lower().replace("sex: ", "")
            assert(sex in ["male", "female"])
            sex = int(sex == "male")

            age = age.lower().replace("age: ", "")
            age = int(age) if age.isdigit() else np.nan
            processed_demos.append([age, sex])
        processed_demos = np.array(processed_demos)
        return processed_demos

    def _split_data(
        self,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[Tuple, Tuple, Tuple]:
        """
        Args:
            data (np.ndarray): Array of shape (num_samples, ).
            labels (np.ndarray): Array of shape (num_samples, num_classes)
            demographics (np.ndarray): Tuple of shape (num_samples, 2 (age, sex)).
        Returns:
            train_data (Tuple):
            valid_data (Tuple):
            test_data (Tuple):
        """
        msss_1 = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=0.2, random_state=self.split_number)

        for train_idx, test_idx in msss_1.split(data, labels):
            pass

        msss_2 = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=0.5, random_state=self.split_number)

        for valid_idx, test_idx in msss_2.split(data[test_idx], labels[test_idx]):
            pass

        X_train, X_valid, X_test =\
            data[train_idx], data[valid_idx], data[test_idx]
        y_train, y_valid, y_test =\
            labels[train_idx], labels[valid_idx], labels[test_idx]

        return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

    def _dump_data(self, X: np.ndarray, y: np.ndarray, datatype: int) -> None:
        """
        Args:
            X (np.ndarray):
            y (np.ndarray):
            datatype (str):
        Returns:
            None
        """
        print(f"Saving {datatype} data ...")
        X.dump(self.save_dir + f'/X_{datatype}_seed{self.split_number}.npy', protocol=4)
        y.dump(self.save_dir + f'/y_{datatype}_seed{self.split_number}.npy', protocol=4)

    def prepare(self):
        """
        Args:

        Returns:

        """
        # Load G12EC data
        signals, dxs, demographics = self._load_data()

        processed_labels, label_index = self._process_label(dxs)
        # processed_demos = self._process_demographics(demographics)

        # Split data into train, valid, test
        (X_train, y_train), (X_val, y_val), (X_test, y_test) =\
            self._split_data(signals, processed_labels)

        X_train, X_val, X_test = self._preprocess_signal(X_train, X_val, X_test)

        self._dump_data(X_train, y_train, "train")
        self._dump_data(X_val, y_val, "val")
        self._dump_data(X_test, y_test, "test")
        label_index.dump(self.save_dir + "/label_index.npy")

if __name__ == "__main__":

    SAMPLING_FREQUENCY = 500
    for seed in range(1, 6):
        print(f"Working on split_number: {seed} ...")
        preparator = G12ECPreparator(SAMPLING_FREQUENCY, seed)
        preparator.prepare()
