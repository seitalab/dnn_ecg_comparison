import os
import sys
import threading

# Execute multiclass classification.

exp1a_targets = {
    "cuda:0": ["resnet1d-18"],
    "cuda:1": ["resnet1d-34"],
    "cuda:2": ["resnet1d-50"],
    "cuda:3": ["lambda_resnet1d18"],
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

node_targets = exp1a_targets

commands = prepare_commands(node_targets)

command_idx = int(sys.argv[1])

thread = threading.Thread(target=execute_command, args=commands[command_idx])
thread.start()
