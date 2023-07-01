import argparse
import datetime
import pickle
from pathlib import Path

import numpy as np

from src.neural_network.data import dict2ndarray


parser = argparse.ArgumentParser()
parser.add_argument("Q_file")
parser.add_argument("--train_filename", default="data/network/processed/train_{}")
parser.add_argument("--test_filename", default="data/network/processed/test_{}")
parser.add_argument("--test_ratio", type=float, default=0.1)
args = parser.parse_args()

# filename
dt = datetime.datetime.now().strftime("%Y%m%d")
args.train_filename = args.train_filename.format(dt)
args.test_filename = args.test_filename.format(dt)
Path(args.train_filename).parent.mkdir(parents=True, exist_ok=True)
Path(args.test_filename).parent.mkdir(parents=True, exist_ok=True)


print("Loading Q-file.")
Q = pickle.load(open(args.Q_file, "rb"))
print("Loading completed.")

print("Converting to ndarray...")
q_arr = dict2ndarray(Q)
np.random.shuffle(q_arr)
split_idx = int(len(q_arr) * (1 - args.test_ratio))
print("Conversion completed.")

q_arr[:split_idx].tofile(args.train_filename)
q_arr[split_idx:].tofile(args.test_filename)
print("Completed.")
