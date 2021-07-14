# root = "/home/nonaka/git/dnn_ecg_comparison"
root = "/export/work/users/nonaka"
# data_root = f"{root}/data"
data_root = f"{root}"
save_dir = "."

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

TASKS = ['all', 'diagnostic', 'subdiagnostic',
         'superdiagnostic', 'form', 'rhythm']

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

# Common
num_epochs = 250
freq = 500
length = 2.5
optimizer = "adam"
eval_every = 5
patience = 5
backbone_out_dim = 256
num_classes_multilabel = {
    "all": 71,
    "diagnostic": 44,
    "subdiagnostic": 23,
    "superdiagnostic": 5,
    "form": 19,
    "rhythm": 12,
    "g12ec": 30,
    "cpsc": 9,
}

# Grid search exeperiment
lr_range = [0.01, 0.001, 0.0001]
bs_range = [64, 128, 256]
gridsearch_result_loc = "./result_gridsearch"

# Multilabel classification setting
selection_criteria = "Mean score"
multilabel_result_loc = "./result_exp1"
MULTILABEL_TASKS = ['all', 'diagnostic', 'subdiagnostic',
                    'superdiagnostic', 'form', 'rhythm', 'g12ec', 'cpsc']

# Multiclass classification setting
multiclass_result_loc = "./result_exp2"
MULTICLASS_TASK = ["mc_AF", "mc_IAVB", "mc_LBBB", "mc_RBBB",
                   "mc_PAC", "mc_PVC", "mc_STD", "mc_STE"]

MULTICLASS_LABELS_INDEX = {
    "Normal": {
        "ptbxl": 46,
        "g12ec": 0,
        "cpsc": 0,
    },
    "AF": {
        "ptbxl": 4,
        "g12ec": 6,
        "cpsc": 1,
    },
    "IAVB": {
        "ptbxl": 0,
        "g12ec": 22,
        "cpsc": 2,
    },
    "LBBB": {
        "ptbxl": 11,
        "g12ec": 20,
        "cpsc": 3,
    },
    "RBBB": {
        "ptbxl": 12,
        "g12ec": 5,
        "cpsc": 4,
    },
    "PAC": {
        "ptbxl": 49,
        "g12ec": 17,
        "cpsc": 5,
    },
    "PVC": {
        "ptbxl": 54,
        "g12ec": 11,
        "cpsc": 6,
    },
    "STD": {
        "ptbxl": 63,
        "g12ec": None,
        "cpsc": 7,
    },
    "STE": {
        "ptbxl": None,
        "g12ec": 27,
        "cpsc": 8,
    },
}

# Model name routings
modelname_routing = {
    # modelname as args: ["python filename", "function name"]
    "resnet1d-18": ["resnet1d", "resnet1d18"],
    "resnet1d-34": ["resnet1d", "resnet1d34"],

    "resnet1d-50": ["resnet1d", "resnet1d50"],
    "resnet1d-101": ["resnet1d", "resnet1d101"],
    "resnet1d-152": ["resnet1d", "resnet1d152"],

    "resnext1d-50": ["resnet1d", "resnext1d50_32x4d"],
    "resnext1d-101": ["resnet1d", "resnext1d101_32x8d"],

    "squeezenet1d-1.0": ["squeezenet1d", "squeezenet1d1_0"],
    "squeezenet1d-1.1": ["squeezenet1d", "squeezenet1d1_1"],

    "senet1d-154": ["senet1d", "senet1d154"],
    "se_resnet1d-50": ["senet1d", "se_resnet1d50"],
    "se_resnet1d-101": ["senet1d", "se_resnet1d101"],
    "se_resnet1d-152": ["senet1d", "se_resnet1d152"],

    "se_resnext1d-50": ["senet1d", "se_resnext1d50_32x4d"],
    "se_resnext1d-101": ["senet1d", "se_resnext1d101_32x4d"],

    "effnet1d_b0": ["effnet1d", "effnet1d_b0"],
    "effnet1d_b1": ["effnet1d", "effnet1d_b1"],
    "effnet1d_b2": ["effnet1d", "effnet1d_b2"],
    "effnet1d_b3": ["effnet1d", "effnet1d_b3"],
    "effnet1d_b4": ["effnet1d", "effnet1d_b4"],
    "effnet1d_b5": ["effnet1d", "effnet1d_b5"],
    "effnet1d_b6": ["effnet1d", "effnet1d_b6"],
    "effnet1d_b7": ["effnet1d", "effnet1d_b7"],
    "effnet1d_b8": ["effnet1d", "effnet1d_b8"],

    # LSTM models
    "lstm_d1_h64": ["bi_lstm", "lstm_d1_h64"],
    "lstm_d1_h128": ["bi_lstm", "lstm_d1_h128"],
    "lstm_d1_h256": ["bi_lstm", "lstm_d1_h256"],
    "lstm_d2_h64": ["bi_lstm", "lstm_d2_h64"],
    "lstm_d2_h128": ["bi_lstm", "lstm_d2_h128"],
    "lstm_d2_h256": ["bi_lstm", "lstm_d2_h256"],
    "lstm_d3_h64": ["bi_lstm", "lstm_d3_h64"],
    "lstm_d3_h128": ["bi_lstm", "lstm_d3_h128"],
    "lstm_d3_h256": ["bi_lstm", "lstm_d3_h256"],

    # Transformer models
    "transformer_d1_h1_dim32l": ["transformer", "transformer_d1_h1_dim32l"],
    "transformer_d1_h1_dim32c": ["transformer", "transformer_d1_h1_dim32c"],

    "transformer_d2_h4_dim64l": ["transformer", "transformer_d2_h4_dim64l"],
    "transformer_d4_h4_dim64l": ["transformer", "transformer_d4_h4_dim64l"],
    "transformer_d8_h4_dim64l": ["transformer", "transformer_d8_h4_dim64l"],
    "transformer_d8_h8_dim256l": ["transformer", "transformer_d8_h8_dim256l"],
    "transformer_d8_h4_dim64c": ["transformer", "transformer_d8_h4_dim64c"],
    "transformer_d2_h8_dim256c": ["transformer", "transformer_d2_h8_dim256c"],
    "transformer_d4_h8_dim256c": ["transformer", "transformer_d4_h8_dim256c"],
    "transformer_d8_h8_dim256c": ["transformer", "transformer_d8_h8_dim256c"],

    # Lambda network models.
    "lambda_resnet1d18": ["lambdanet1d", "lambda_resnet1d18"],
    "lambda_resnet1d50": ["lambdanet1d", "lambda_resnet1d50"],
    "lambda_resnet1d101": ["lambdanet1d", "lambda_resnet1d101"],
    "lambda_resnet1d152": ["lambdanet1d", "lambda_resnet1d152"],

    # MobileNetV3
    "mobilenetv3-l": ["mobilenetv3_1d", "mobilenet_v3_large"],
    "mobilenetv3-s": ["mobilenetv3_1d", "mobilenet_v3_small"],

    # NFNet
    "nfresnet1d18": ["nfnet1d", "nf_resnet1d18"],
    "nfresnet1d34": ["nfnet1d", "nf_resnet1d34"],
    "nfresnet1d50": ["nfnet1d", "nf_resnet1d50"],
    "nfresnet1d101": ["nfnet1d", "nf_resnet1d101"],
}
