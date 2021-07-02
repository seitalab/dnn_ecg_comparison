import os
import pickle
from typing import Optional, List, Tuple

import numpy as np
from torch.utils.data import Dataset

class PTBXLDataset(Dataset):

    def __init__(self, root: str, datatype: str,
                 data_split: str, transform:Optional[List]=None):
        """
        Args:
            root (str): Path to dataset directory.
            datatype (str): Dataset type to load (train, valid, test)
            data_split (str): String of `val-XX_test-XX`.
            transform (List): List of transformations to be applied.
        """
        assert(datatype in ["train", "val", "test"])

        self.data_loc = os.path.join(root, data_split)

        self.data, self.label = self._load_data(datatype)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data, label = self.data[index], self.label[index]

        if self.transform:
            sample = {"data": data, "label": label}
            sample = self.transform(sample)
            data, label = sample["data"], sample["label"]

        return data, label

    def _load_data(self, datatype: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load npy file of target datatype.

        Args:
            datatype (str)
        Returns:
            X (np.ndarray): Array of shape (num_samples, 12, sequence_length).
            y (np.ndarray): Array of shape (num_samples, ??).
        """
        Xfile = os.path.join(self.data_loc, f"X_{datatype}.npy")
        yfile = os.path.join(self.data_loc, f"y_{datatype}.npy")
        X = np.load(Xfile, allow_pickle=True)
        y = np.load(yfile, allow_pickle=True)
        X = np.transpose(X, (0, 2, 1))
        return X, y

class G12ECDataset(PTBXLDataset):

    def __init__(self, root: str, datatype: str,
                 data_split: int, transform:Optional[List]=None):
        """
        Args:
            root (str): Path to dataset directory.
            datatype (str): Dataset type to load (train, valid, test)
            data_split (int): Seed number.
            transform (List): List of transformations to be applied.
        """
        assert(datatype in ["train", "val", "test"])

        self.data_loc = root

        self.data, self.label = self._load_data(datatype, data_split)

        self.transform = transform

    def _load_data(
        self,
        datatype: str,
        data_split: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load npy file of target datatype.

        Args:
            datatype (str): Dataset type to load (train, valid, test)
            data_split (int): Seed number.
        Returns:
            X (np.ndarray): Array of shape (num_samples, 12, sequence_length).
            y (np.ndarray): Array of shape (num_samples, num_classes=??).
        """
        Xfile = os.path.join(self.data_loc, f"X_{datatype}_seed{data_split}.npy")
        yfile = os.path.join(self.data_loc, f"y_{datatype}_seed{data_split}.npy")
        X = np.load(Xfile, allow_pickle=True).astype(float)
        y = np.load(yfile, allow_pickle=True).astype(int)
        X = np.transpose(X, (0, 2, 1))
        return X, y

class CPSCDataset(G12ECDataset):

    def _load_data(
        self,
        datatype: str,
        data_split: int
    ) -> Tuple[List, np.ndarray]:
        """
        Load pkl and npy file of target datatype.

        Args:
            datatype (str): Dataset type to load (train, valid, test)
            data_split (int): Seed number.
        Returns:
            X (List): List of arrays (num_samples, ).
            y (np.ndarray): Array of shape (num_samples, num_classes=??).
        """
        Xfile = os.path.join(self.data_loc, f"X_{datatype}_seed{data_split}.pkl")
        with open(Xfile, "rb") as fp:
            X = pickle.load(fp)
        yfile = os.path.join(self.data_loc, f"y_{datatype}_seed{data_split}.npy")
        y = np.load(yfile, allow_pickle=True).astype(int)
        return X, y

if __name__ == "__main__":
    data_loc = "/home/nonaka/mnt/PTBXL/all/"
    data_split = "val-9_test-10"
    dataset = PTBXLDataset(data_loc, "val", data_split)
    print(dataset[0])
    print(dataset.data.shape)
    print(dataset.label.shape)
