import sys
from typing import Type

from torchvision import transforms
from torch.utils.data import DataLoader

sys.path.append("..")
import config
from codes.data.dataset import PTBXLDataset, G12ECDataset, CPSCDataset
from codes.data.transform_funcs import (
    Subsample, SubsampleEval, ToTensor, ProcessLabel
)

def form_datasplit_string(split_number: int) -> str:
    """
    Form string containing datasplit information (eg. `val-9_test-10`)

    Args:
        split_number (int):
    Returns:
        data_split_string (str):
    """
    fold_indices = config.split_settings[split_number]
    val_fold_index = fold_indices["val_index"]
    test_fold_index = fold_indices["test_index"]
    data_split_string = f"val-{val_fold_index}_test-{test_fold_index}"
    return data_split_string

def prepare_preprocess(
    frequency: int,
    length: int,
    is_train: bool
) -> Type[transforms.Compose]:
    """
    Prepare and compose transform functions.
    Args:
        frequency (int):
        length (int):
        is_train (bool):
    Returns:
        composed
    """
    subsample_length = int(frequency * length)
    if is_train:
        composed = transforms.Compose(
            [Subsample(subsample_length), ToTensor()])
    else:
        composed = transforms.Compose(
            [SubsampleEval(subsample_length), ToTensor()])
    return composed

def prepare_dataloader(
    task_name: str,
    data_loc: str,
    datatype: str,
    batch_size: int,
    split_number: int,
    frequency: int,
    length: int,
    is_train: bool
) -> Type[DataLoader]:
    """
    Args:
        task_name (str): Name of dataset ("all", "diagnostic", .., "g12ec").
        data_loc (str): Path to data pkl file.
        datatype (str): Type of dataset ("train", "val", "test").
        batch_size (int): batch size
        split_number (int):
        frequency (int):
        length (int):
        is_train (bool):
    Returns:
        Dataloader (dataloader):
    """
    transformations = prepare_preprocess(frequency, length, is_train)
    if task_name in config.TASKS:
        data_split_string = form_datasplit_string(split_number)
        dataset = PTBXLDataset(
            data_loc, datatype, data_split_string, transformations)
    elif task_name == "g12ec":
        dataset = G12ECDataset(
            data_loc, datatype, split_number, transformations)
    elif task_name == "cpsc":
        dataset = CPSCDataset(
            data_loc, datatype, split_number, transformations)
        # batch_size of cpsc dataset on eval mode needs to be 1.
        batch_size = 1 if not is_train else batch_size
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=is_train, drop_last=is_train,
                        worker_init_fn=split_number)
    return loader

def prepare_preprocess_multiclass(
    frequency: int,
    length: int,
    normal_index: int,
    target_index: int,
    is_train: bool
) -> Type[transforms.Compose]:
    """
    Prepare and compose transform functions.
    Args:
        frequency (int):
        length (int):
        target_index (int):
        normal_index (int):
        is_train (bool):
    Returns:
        composed
    """
    subsample_length = int(frequency * length)
    if is_train:
        composed = transforms.Compose([
            Subsample(subsample_length),
            ProcessLabel(normal_index, target_index),
            ToTensor()
        ])
    else:
        composed = transforms.Compose([
            SubsampleEval(subsample_length),
            ProcessLabel(normal_index, target_index),
            ToTensor()
        ])
    return composed

def prepare_dataloader_multiclass(
    task_name: str,
    dataset_name: str,
    data_loc: str,
    datatype: str,
    batch_size: int,
    split_number: int,
    frequency: int,
    length: int,
    is_train: bool
) -> Type[DataLoader]:
    assert(dataset_name in config.DATASETS)
    assert(task_name.startswith("mc_")) # eg. `mc_AF`
    target_dx = task_name[3:]
    normal_index = config.MULTICLASS_LABELS_INDEX["Normal"][dataset_name]
    target_index = config.MULTICLASS_LABELS_INDEX[target_dx][dataset_name]

    transformations = prepare_preprocess_multiclass(
        frequency, length, normal_index, target_index, is_train)

    if dataset_name == "ptbxl":
        data_split_string = form_datasplit_string(split_number)
        dataset = PTBXLDataset(
            data_loc, datatype, data_split_string, transformations)
    elif dataset_name == "g12ec":
        dataset = G12ECDataset(
            data_loc, datatype, split_number, transformations)
    elif dataset_name == "cpsc":
        dataset = CPSCDataset(
            data_loc, datatype, split_number, transformations)
        # batch_size of cpsc dataset on eval mode needs to be 1.
        batch_size = 1 if not is_train else batch_size
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=is_train, drop_last=is_train,
                        worker_init_fn=split_number)
    return loader
