import os
import sys
import threading

# Execute grid search

exp0a_targets = {
    "cuda:0": ["resnet1d-18", "resnext1d-101", "effnet1d_b0", "effnet1d_b2"],
    "cuda:1": ["resnet1d-34", "se_resnext1d-101", "effnet1d_b1", "effnet1d_b3"],
    "cuda:2": ["resnet1d-50", "se_resnet1d-101", "resnext1d-50", "effnet1d_b4"],
    "cuda:3": ["resnet1d-101", "se_resnext1d-50", "se_resnet1d-50", "effnet1d_b5"],
}

def prepare_commands(targets):
    commands = []
    for device, modelnames in targets.items():
        device_commands = []
        for modelname in modelnames:
            command = f"python execute_grid_search.py {modelname} {device}"
            device_commands.append(command)
        commands.append(device_commands)
    return commands

def execute_command(*commands):
    for i in range(len(commands)):
        os.system(commands[i])

node_targets = exp0a_targets
commands = prepare_commands(node_targets)

command_idx = int(sys.argv[1])

thread = threading.Thread(target=execute_command, args=commands[command_idx])
thread.start()
