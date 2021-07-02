import numpy as np
import torch

np.random.seed(0)

class Subsample(object):
    """
    Subsample fixed length of ECG signals.

    Args:
        subsample_length (int): Length of subsampled data.
    """
    def __init__(self, subsample_length: int):

        assert isinstance(subsample_length, int)
        self.subsample_length = subsample_length

    def __call__(self, sample):
        """
        Args:
            sample (Dict): {"data": Array of shape (12, sequence_length).,
                            "label": label}
        Returns:
            sample (Dict): {"data": Array of shape (12, subsample_length).,
                            "label": label}
        """
        data, label = sample["data"], sample["label"]

        start = np.random.randint(0, data.shape[1] - self.subsample_length)
        subsampled_data = data[:, start:start+self.subsample_length]

        return {"data": subsampled_data, "label": label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    # def __init__(self, label_type: str="float"):
    #     self.label_type = label_type

    def __call__(self, sample):
        data, label  = sample["data"], sample["label"]
        data_tensor = torch.from_numpy(data)
        label_tensor = torch.from_numpy(label)
        # data_tensor = data_tensor.float()
        # if self.label_type == "float":
        #     label_tensor = label_tensor.float()
        # elif self.label_type == "long":
        #     label_tensor = label_tensor.long()
        # else:
        #     raise NotImplementedError
        return {"data": data_tensor, "label": label_tensor}

class SubsampleEval(Subsample):
    """
    Subsampling for evaluation mode.

    Args:
        subsample_length (int): Length of subsampled data.
    """

    def _pad_signal(self, data):
        """
        Args:
            data (np.ndarray):
        Returns:
            padded_data (np.ndarray):
        """
        chunk_length = self.subsample_length // 2
        pad_length = chunk_length - data.shape[1] % chunk_length

        if pad_length == 0:
            return data
        pad = np.zeros([12, pad_length])
        pad_data = np.concatenate([data, pad], axis=-1)
        return pad_data

    def __call__(self, sample):
        """
        Args:
            sample (Dict): {"data": Array of shape (12, sequence_length).,
                            "label": label}
        Returns:
            sample (Dict): {"data": Array of shape (12, num_split, subsample_length).,
                            "label": label}
        """
        data, label = sample["data"], sample["label"]
        slice_indices = np.arange(0, data.shape[1], self.subsample_length // 2)
        index_range = np.arange(self.subsample_length)
        target_locs = slice_indices[:, np.newaxis] + index_range[np.newaxis]

        padded_data = self._pad_signal(data)
        try:
            eval_subsamples = padded_data[:, target_locs]
        except:
            eval_subsamples = padded_data[:, target_locs[:-1]]
        return {"data": eval_subsamples, "label": label}

class ProcessLabel(object):
    "Convert to multiclass label"

    def __init__(
        self,
        normal_index: int,
        target_index: int,
        num_classes: int = 3
    ) -> None:
        self.normal_index = normal_index
        self.target_index = target_index
        self.num_classes = num_classes

    def __call__(self, sample):
        """
        Args:
            sample (Dict): {"data": Array of shape (12, sequence_length).,
                            "label": label}
        Returns:
            sample (Dict): {"data": Array of shape (12, num_split, subsample_length).,
                            "label": label}
        """
        data, label = sample["data"], sample["label"]

        if label[self.target_index]:
            processed_label = 1
        elif label[self.normal_index]:
            processed_label = 0
        else:
            processed_label = 2
        processed_label = np.array(processed_label)

        return {"data": data, "label": processed_label}
