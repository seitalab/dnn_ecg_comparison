import os
import sys
import pickle
from typing import List, Tuple
from glob import glob

import numpy as np
import pandas as pd
from scipy import io
from tqdm import tqdm

sys.path.append("..")
import config
from g12ec import G12ECPreparator

class CPSCPreparator(G12ECPreparator):

    def __init__(self,
        sampling_frequency: int=500,
        split_number: int=1,
    ) -> None:
        """
        Args:
            sampling_frequency (int): Sampling frequency (500).
            split_number (int): Seed value for train test split.
        """

        self.sampling_frequency = sampling_frequency
        self.split_number = split_number

        self.load_dir = os.path.join(
            config.data_root, config.dirname_cpsc, "raw")
        self.save_dir = os.path.join(
            config.data_root, config.dirname_cpsc, "processed_modelclf")
        os.makedirs(self.save_dir, exist_ok=True)

    def _open_ecg_files(self, target_files: List) -> List[np.ndarray]:
        """
        Load signal and extract signal information of given file name.
        Args:
            target_files (List): List of name of target file to load
        Returns:
            12lead_signal (ndarray): numpy array of 12 leads
        """
        signals = []
        for target_file in tqdm(target_files):
            target_path = os.path.join(self.load_dir, "*", target_file+".mat")
            mat_file = glob(target_path)[0]
            record = io.loadmat(mat_file) # [`Sex`, `Age`, `ECG`]
            signals.append(record["ECG"][0][0][2]) # num_lead = 12, sequence_length
        return signals

    def _load_filelist(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load list of files and labels in dataset.

        Args:
            None
        Returns:
            files: ndarray of signal file names
            labels: ndarray of labels
        """
        reference_file = os.path.join(self.load_dir, config.cpsc_reference)
        filelist_df = pd.read_csv(reference_file)
        files = filelist_df.Recording.values
        labels = self._process_label(filelist_df)

        return files, labels

    def _process_label(self, df: pd.DataFrame) -> np.ndarray:
        """
        Convert dataframe of label to np.ndarray of shape [num_labels = 9].
        Args:
            df (pd.DataFrame): DataFrame of labels (contains maximum of 3 labels.)
        Returns:
            labels (np.ndarray): Array of shape [num_labels = 9], each element is a binary.
        """
        labels = np.zeros([df.shape[0], 9])

        labels_first = df.First_label.values
        labels_second = df.Second_label.values
        labels_third = df.Third_label.values

        # labels = [1 - 9] -> label_index = [0 - 8]
        for i in range(df.shape[0]):
            # Validate: normal labeled sample has no other labels.
            if labels_first[i] == 1:
                assert(np.isnan(labels_second[i]) \
                       and np.isnan(labels_third[i]))

            labels[i, labels_first[i] - 1] = 1
            if not np.isnan(labels_second[i]):
                labels[i, int(labels_second[i]) - 1] = 1
            if not np.isnan(labels_third[i]):
                labels[i, int(labels_third[i]) - 1] = 1
        return labels

    def _dump_data(self, X: List, y: np.ndarray, datatype: int) -> None:
        """
        Args:
            X (np.ndarray):
            y (np.ndarray):
            datatype (str):
        Returns:
            None
        """
        print(f"Saving {datatype} data ...")
        savename = self.save_dir + f'/X_{datatype}_seed{self.split_number}.pkl'
        with open(savename, "wb") as fp:
            pickle.dump(X, fp)
        y.dump(self.save_dir + f'/y_{datatype}_seed{self.split_number}.npy', protocol=4)

    def prepare(self) -> None:
        """
        Args:

        Returns:

        """
        # Load CPSC data
        files, labels = self._load_filelist()

        # Split data into train, valid, test
        (f_train, y_train), (f_val, y_val), (f_test, y_test) =\
            self._split_data(files, labels)

        X_train = self._open_ecg_files(f_train)
        X_val = self._open_ecg_files(f_val)
        X_test = self._open_ecg_files(f_test)

        self._dump_data(X_train, y_train, "train")
        self._dump_data(X_val, y_val, "val")
        self._dump_data(X_test, y_test, "test")

if __name__ == "__main__":

    SAMPLING_FREQUENCY = 500
    for seed in range(1, 6):
        print(f"Working on split_number: {seed} ...")
        preparator = CPSCPreparator(SAMPLING_FREQUENCY, seed)
        preparator.prepare()
