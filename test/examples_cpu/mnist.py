import torch
import scorch
import torchvision.datasets
import scorch.base
import scorch.callbacks
import os
import numpy
import sys

sys.path.append(os.getcwd())

from datasets import MNIST
from models import ThreeLayersFullyConnected, LeNetFullyConvolutional

ds = MNIST()

## Trying fully connected

net = ThreeLayersFullyConnected(n = [100, 100, 10])

def loss(predictions, targets):
    return torch.nn.functional.cross_entropy(
        predictions[0], targets[0])

def acc(predictions, targets):
    return (predictions[0].argmax(dim=-1) == targets[0]).sum(), targets[0].numel()

trainer = scorch.base.Trainer(
    net,
    criterion=loss,
    optimizers=[scorch.base.OptimizerSwitch(net, torch.optim.Adam, lr=3.0e-4)],
    callbacks=[scorch.callbacks.ComputeMetrics([loss, acc]),
               scorch.callbacks.ExponentialWeightAveraging(epoch_start=2)]
)

for index in range(5):
    trainer.train_one_epoch(ds, subset='train', batch_size=128)
    trainer.validate_one_epoch(ds, subset='train', batch_size=128)
    trainer.validate_one_epoch(ds, subset='valid', batch_size=128)

## Trying fully connected

net = ThreeLayersFullyConnected(n = [100, 100, 10], batch_norms=True)

def loss(predictions, targets):
    return torch.nn.functional.cross_entropy(
        predictions[0], targets[0])

def acc(predictions, targets):
    return (predictions[0].argmax(dim=-1) == targets[0]).sum(), targets[0].numel()

trainer = scorch.base.Trainer(
    net,
    criterion=loss,
    optimizers=[scorch.base.OptimizerSwitch(net, torch.optim.Adam, lr=3.0e-4)],
    callbacks=[scorch.callbacks.ComputeMetrics([loss, acc]),
               scorch.callbacks.ExponentialWeightAveraging(epoch_start=2)]
)

for index in range(5):
    trainer.train_one_epoch(ds, subset='train', batch_size=128)
    trainer.validate_one_epoch(ds, subset='train', batch_size=128)
    trainer.validate_one_epoch(ds, subset='valid', batch_size=128)


## Trying fully convolutional

# net = LeNetFullyConvolutional(channels=[1, 64, 128, 128, 10])
#
#
# def loss(predictions, targets):
#     return torch.nn.functional.cross_entropy(
#         predictions[0], targets[0])
#
# def acc(predictions, targets):
#     return (predictions[0].argmax(dim=-1) == targets[0]).sum(), targets[0].numel()
#
# trainer = scorch.base.Trainer(
#     net,
#     criterion=loss,
#     optimizers=[scorch.base.OptimizerSwitch(net, torch.optim.Adam, lr=3.0e-4)],
#     callbacks=[scorch.callbacks.ComputeMetrics([loss, acc]),
#                scorch.callbacks.ExponentialWeightAveraging(epoch_start=2)]
# )
#
# for index in range(5):
#     trainer.train_one_epoch(ds, subset='train', batch_size=128)
#     trainer.validate_one_epoch(ds, subset='train', batch_size=128)
#     trainer.validate_one_epoch(ds, subset='valid', batch_size=128)
