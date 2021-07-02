root = "/home/nonaka/git/dnn_ecg_comparison"
data_root = f"{root}/data"

dirname_ptbxl = "PTBXL"
dirname_g12ec = "G12EC"
dirname_cpsc = "CPSC2018"
DATASETS = ["cpsc", "g12ec", "ptbxl"]

# G12EC
g12ec_lead_order = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
g12ec_default_signal_length = 5000 # Value from document

# CPSC
cpsc_reference = "TrainingSet3/REFERENCE.csv"
cpsc_dxs = ["Normal", "AF", "IAVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]
lead_order = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# PTB-XL
split_settings = {
    1: {
        "val_index": 9,
        "test_index": 10
    },
    2: {
        "val_index": 7,
        "test_index": 8,
    },
    3: {
        "val_index": 5,
        "test_index": 6,
    },
    4: {
        "val_index": 3,
        "test_index": 4,
    },
    5: {
        "val_index": 1,
        "test_index": 2,
    },
}
