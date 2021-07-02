import os
import sys
import threading

# Execute multiclass classification.

exp1a_targets = {
    #"cuda:0": ["resnet1d-18"],
    # "cuda:0": ["mobilenetv3-l"],
    "cuda:0": ["resnext1d-50"],
    "cuda:1": ["se_resnext1d-50"],
    "cuda:2": ["se_resnet1d-50"],
    "cuda:3": [],
}

exp1b_targets = {
    "cuda:0": ["resnet1d-18"],
    "cuda:1": ["resnet1d-34"],
    "cuda:2": ["resnet1d-50"],
    "cuda:3": ["lambda_resnet1d18"],
}

exp1c_targets = {
    "cuda:0": ["effnet1d_b0"],
    "cuda:1": ["effnet1d_b1"]
}

exp1d_targets = {
    "cuda:0": ["transformer_d2_h4_dim64l"],
    "cuda:1": ["transformer_d4_h4_dim64l"],
    "cuda:2": ["lstm_d1_h64"],
    "cuda:3": ["lambda_resnet1d50"],
}


exp1e_targets = {
    "cuda:0": ["mobilenetv3-s"],
    "cuda:1": ["nfresnet1d18"],
    "cuda:2": ["nfresnet1d34"],
    "cuda:3": ["nfresnet1d50"],
}

def prepare_commands(targets):
    commands = []
    for device, modelnames in targets.items():
        device_commands = []
        for modelname in modelnames:
            command = f"python execute_clf_multilabel.py {modelname} {device}"
            device_commands.append(command)
        commands.append(device_commands)
    return commands

def execute_command(*commands):
    for i in range(len(commands)):
        os.system(commands[i])
        # print(commands[i])

# node_targets = exp1a_targets
# node_targets = exp1b_targets
# node_targets = exp1c_targets
# node_targets = exp1d_targets
node_targets = exp1e_targets

commands = prepare_commands(node_targets)

command_idx = int(sys.argv[1])

thread = threading.Thread(target=execute_command, args=commands[command_idx])
thread.start()
