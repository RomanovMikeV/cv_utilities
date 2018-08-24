import torch
import torch.utils.data.distributed
import torchvision
import skimage.transform
import sys
import argparse
import time
import importlib.util
import horovod.torch as hvd
import gc
from tensorboardX import SummaryWriter

import torchvision.transforms as transform

from . import trainer
from . import utils


def train(model_source_path,
          dataset_source_path,
          dataset_path,
          batch_size=4,
          num_workers=0,
          dump_period=1,
          epochs=1000,
          checkpoint=None,
          use_cuda=False,
          validate_on_train=False,
          max_train_iterations=-1,
          max_valid_iterations=-1,
          verbosity=-1,
          checkpoint_prefix='',
          dataset_kwargs={},
          model_kwargs={}):

    # Initializing Horovod
    hvd.init()

    # Enabling garbage collector
    gc.enable()

    # Creating tensorboard writer
    if hvd.rank() == 0:
        tb_writer = SummaryWriter()

    spec = importlib.util.spec_from_file_location("model", model_source_path)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)

    spec = importlib.util.spec_from_file_location("dataset", dataset_source_path)
    dataset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset)

    ## Creating neural network and a socket for it

    net = model.Network(**model_kwargs)
    #net = torch.nn.parallel.DataParallel(net).eval()

    if use_cuda:
        torch.cuda.set_device(hvd.local_rank())
        net = net.cuda()

    socket = model.Socket(net)
    socket.optimizer = hvd.DistributedOptimizer(
        socket.optimizer,
        named_parameters=socket.model.named_parameters())

    hvd.broadcast_parameters(socket.model.state_dict(), root_rank=0)

    # Creating the datasets

    ds_index = dataset.DataSetIndex(dataset_path, **dataset_kwargs)

    train_dataset = dataset.DataSet(
        ds_index, mode='train')

    valid_dataset = dataset.DataSet(
        ds_index, mode='valid')

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dataset, num_replicas=hvd.size(), rank=hvd.rank())

    ## Creating dataloaders from the datasets

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=num_workers,
                                               drop_last=True,
                                               pin_memory=False,
                                               collate_fn=utils.default_collate,
                                               sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=num_workers,
                                               drop_last=False,
                                               pin_memory=False,
                                               collate_fn=utils.default_collate,
                                               sampler=valid_sampler)

    if hvd.rank() != 0:
        verbosity = -1

    my_trainer = trainer.Trainer(socket,
                                 verbosity=verbosity,
                                 use_cuda=use_cuda,
                                 max_train_iterations=max_train_iterations,
                                 max_valid_iterations=max_valid_iterations)
    best_metrics = None

    if checkpoint is not None:
        my_trainer = trainer.load_from_checkpoint(checkpoint,
                                                  socket,
                                                  verbosity=verbosity,
                                                  use_cuda=use_cuda,
                                                  max_train_iterations=max_train_iterations,
                                                  max_valid_iterations=max_valid_iterations)
        best_metrics = my_trainer.metrics


    for epoch_index in range(epochs):
        #try:
        loss = my_trainer.train(train_loader)
        gc.collect()
        #print("Problem in train")

        train_metrics = {}

        if validate_on_train:
            train_metrics = my_trainer.validate(train_loader)
            gc.collect()

        valid_metrics = my_trainer.validate(valid_loader)
        for metric_name in valid_metrics:
            metric_data = {'valid': valid_metrics[metric_name]}

            if metric_name in train_metrics:
                metric_data['train'] = train_metrics[metric_name]

            if hvd.rank() == 0:
                tb_writer.add_scalars('metrics/' + metric_name + '/' + checkpoint_prefix, metric_data, my_trainer.epoch)
        #print(epoch_index, '<- epoch index')

        gc.collect()
        #except:
        #    print("Problem in validation")
        #    metrics = my_trainer.metrics

        socket.scheduler.step(valid_metrics['main'])
        gc.collect()

        if epoch_index % dump_period == 0 and hvd.rank() == 0:
            is_best = False
            if best_metrics == None:
                is_best = True
            else:
                if best_metrics['main'] < my_trainer.metrics['main']:
                    is_best = True
            if is_best:
                best_metrics = my_trainer.metrics

            my_trainer.make_checkpoint(checkpoint_prefix,
                                       info=valid_metrics,
                                       is_best=is_best)
        gc.collect()


def training():

    ## Training parameters

    parser = argparse.ArgumentParser(
        description='Script to train a specified model with a specified dataset.')
    parser.add_argument('-b','--batch-size', help='Batch size', default=8, type=int)
    parser.add_argument('-w','--workers', help='Number of workers in a dataloader', default=0, type=int)
    parser.add_argument('-d', '--dump-period', help='Dump period', default=1, type=int)
    parser.add_argument('-e', '--epochs', help='Number of epochs to perform', default=1000, type=int)
    parser.add_argument('-c', '--checkpoint', help='Checkpoint to load from', default=None)
    parser.add_argument('--use-cuda', help='Use cuda for training', action='store_true')
    parser.add_argument('--validate-on-train', help='Validate on train', action='store_true')
    parser.add_argument('--model', help='File with a model specifications', required=True)
    parser.add_argument('--dataset', help='File with a dataset sepcification', required=True)
    parser.add_argument('--max-train-iterations', help='Maximum training iterations', default=-1, type=int)
    parser.add_argument('--max-valid-iterations', help='Maximum validation iterations', default=-1, type=int)
    parser.add_argument('-dp', '--dataset-path', help='Path to the dataset', required=True)
    parser.add_argument('-v', '--verbosity',
                        help='-1 for no output, 0 for epoch output, positive number is printout frequency',
                        default=-1, type=int)
    parser.add_argument('-cp', '--checkpoint-prefix', help='Prefix to the checkpoint name', default='')
    parser.add_argument('--model-args', help='Model arguments which will be used during training', default='', type=str)
    parser.add_argument('--dataset-args', help='Dataset arguments which will be used during training', default='', type=str)
    #parser.add_argument('--xorovod')
    args = vars(parser.parse_args())

    ## Calling training function

    train(args['model'], args['dataset'], args['dataset_path'],
              batch_size=args['batch_size'],
              num_workers=args['workers'],
              dump_period=args['dump_period'],
              epochs=args['epochs'],
              checkpoint=args['checkpoint'],
              use_cuda=args['use_cuda'],
              validate_on_train=args['validate_on_train'],
              max_train_iterations=args['max_train_iterations'],
              max_valid_iterations=args['max_valid_iterations'],
              verbosity=args['verbosity'],
              checkpoint_prefix=args['checkpoint_prefix'],
              dataset_kwargs=eval('{' + args['dataset_args'] + '}'),
              model_kwargs=eval('{' + args['model_args'] + '}'))

if __name__ == '__main__':
    training()