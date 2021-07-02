import argparse

parser = argparse.ArgumentParser()

# Common
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--model', type=str, default="resnet1d-18")
parser.add_argument('--task', type=str, default="all")
parser.add_argument('--freq', type=int, default=500)
parser.add_argument('--length', type=float, default=2.5)
parser.add_argument('--backbone_out_dim', type=int, default=256)

# NN
parser.add_argument('--ep', type=int, default=10)
parser.add_argument('--bs', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--optim', type=str, default="adam")
parser.add_argument('--device', type=str, default="cpu")
parser.add_argument('--eval-every', type=int, default=5)
parser.add_argument('--patience', type=int, default=5)

args = parser.parse_args()
args.split_number = args.seed # Align seed number and data split number (for simplisity)
args.sequence_length = args.freq * args.length

print(args)
if __name__ == "__main__":
    print("-"*80)
    print(args)
