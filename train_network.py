import argparse
import json
from logging import getLogger
from logging.config import dictConfig
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# from src.neural_network.data import MultiLabelDataset
# from src.neural_network.network import Model
from src.neural_network.data import MultiLabelDatasetOneChannel as MultiLabelDataset
from src.neural_network.network_02 import Model

parser = argparse.ArgumentParser()
parser.add_argument("train_data", type=str, help="training data file")
parser.add_argument("test_data", type=str, help="test data file")
parser.add_argument("--gpu", "-g", type=int, default=0, help="GPU ID")
parser.add_argument("--epoch", "-e", type=int, default=1, help="Number of epoch times")
parser.add_argument("--batchsize", "-b", type=int, default=1024,
                    help="Number of positions in each mini-batch")
parser.add_argument("--test_batchsize", type=int, default=1024,
                    help="Number of positions in each test mini-batch")
parser.add_argument("--lr", type=float, default=0.01, help="learnung rate")
parser.add_argument("--checkpoint", default="checkpoints/checkpoint-{epoch:03}.pth",
                    help="checkpoint file name")
parser.add_argument("--resume", "-r", default="", help="resume from snapshot")
parser.add_argument("--eval_interval", type=int, default=100, help="evaluation interval")
parser.add_argument("--log", default="log/nn_train.log", help="log file path")
args = parser.parse_args()

# load logger
conf_file = "config/logger.json"
log_conf = json.load(open(conf_file))
log_conf["handlers"]["fileHandler"]["filename"] = args.log
dictConfig(log_conf)
logger = getLogger("simpleLogger")
logger.info(f"batchsize={args.batchsize}, lr={args.lr}")

# device
if args.gpu >= 0:
    device = torch.device(f"cuda:{args.gpu}")
else:
    device = torch.device("cpu")
logger.info(f"Device = {device}")

# model
model = Model()
model.to(device)

# optimizer
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)

# loss function. Policy and Value use the same function.
bce_loss = torch.nn.BCELoss()

# load checkpoint
if args.resume:
    logger.info(f"Loading the checkpoint from {args.resume}")
    checkpoint = torch.load(args.resume, map_location=device)
    epoch = checkpoint["epoch"]
    t = checkpoint["t"]  # total steps
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # change learning rate to argument value
    optimizer.param_groups[0]["lr"] = args.lr
else:
    epoch = 0
    t = 0  # total steps

# read train & test data
logger.info("reading training and test data.")
train_datasets = MultiLabelDataset(args.train_data)
test_datasets = MultiLabelDataset(args.test_data)
train_dataloader = DataLoader(
    train_datasets,
    batch_size=args.batchsize,
    shuffle=True,
    num_workers=5,
    drop_last=True,
    pin_memory=True if device != torch.device("cpu") else False
)
test_dataloader = DataLoader(
    test_datasets,
    batch_size=args.test_batchsize,
    shuffle=False,
    num_workers=3,
    drop_last=True,
    pin_memory=True if device != torch.device("cpu") else False
)
logger.info(f"Train data num: {len(train_datasets):,}")
logger.info(f"Test data num : {len(test_datasets):,}")


def binary_accuracy(y, t):
    """calculate policy and value accuracy"""
    pred = (y >= 0)
    truth = (t >= 0.5)
    return pred.eq(truth).sum().item() / len(t)

def save_checkpoint():
    path = args.checkpoint.format(**{"epoch": epoch, "step": t})
    Path(path).parent.mkdir(exist_ok=True)
    logger.info(f"Saving the checkpoint to {path}")
    checkpoint = {
        "epoch": epoch,
        "t": t,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(checkpoint, path)

# train loop
for e in range(args.epoch):
    epoch += 1
    steps_interval = 0
    sum_loss_policy_interval = 0
    sum_loss_value_interval = 0
    steps_epoch = 0
    sum_loss_policy_epoch = 0
    sum_loss_value_epoch = 0
    for x, label, value in train_dataloader:
        model.train()
        x = x.to(device)
        label = label.to(device)
        value = value.to(device)

        # forward
        y1, y2 = model(x)
        # calc loss
        loss_policy = bce_loss(y1, label)
        loss_value = bce_loss(y2, value)
        loss = loss_policy + loss_value
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # add 1 to total step
        t += 1

        # Add 1 to step counter, add losses to total loss for evaluation.
        # for interval
        steps_interval += 1
        sum_loss_policy_interval += loss_policy.item()
        sum_loss_value_interval += loss_value.item()
        # for epoch
        steps_epoch += 1
        sum_loss_policy_epoch += loss_policy.item()
        sum_loss_value_epoch += loss_value.item()

        # Display training loss and test loss and accuracy
        # for each evaluation interval.
        if t % args.eval_interval == 0:
            logger.info(
                f"epoch = {epoch}, step = {t}, train loss = "
                f"{sum_loss_policy_interval / steps_interval:.3f}, "
                f"{sum_loss_value_interval / steps_interval:.3f}, "
                f"{(sum_loss_value_interval + sum_loss_value_interval) / steps_interval:.3f} "
            )

        steps_interval = 0
        sum_loss_policy_interval = 0
        sum_loss_value_interval = 0

    # Evaluate using all of the data at the end of the epoch.
    test_steps = 0
    sum_test_loss_policy = 0
    sum_test_loss_value = 0
    sum_test_accuracy_policy = 0
    sum_test_accuracy_value = 0

    model.eval()
    with torch.no_grad():
        for x, label, value in test_dataloader:
            x = x.to(device)
            label = label.to(device)
            value = value.to(device)

            y1, y2 = model(x)

            test_steps += 1
            sum_test_loss_policy += bce_loss(y1, label).item()
            sum_test_loss_value += bce_loss(y2, value).item()
            sum_test_accuracy_policy += binary_accuracy(y1, label)
            sum_test_accuracy_value += binary_accuracy(y2, value)

    logger.info(
        f"epoch = {epoch}, step = {t}, "
        "train loss = "
        f"{sum_loss_policy_epoch / steps_epoch:.3f}, "
        f"{sum_loss_value_epoch / steps_epoch:.3f}, "
        f"{(sum_loss_value_epoch + sum_loss_value_epoch) / steps_epoch:.3f} "
        f"test loss= {sum_test_loss_policy / test_steps:.3f}, "
        f"{sum_test_loss_value / test_steps:.3f}, "
        f"{(sum_test_loss_policy + sum_test_loss_value) / test_steps:.3f}, "
        f"test accuracy = {sum_test_accuracy_policy / test_steps:.3f}, "
        f"{sum_test_accuracy_value / test_steps:.3f}"
    )

    if args.checkpoint:
        save_checkpoint()
