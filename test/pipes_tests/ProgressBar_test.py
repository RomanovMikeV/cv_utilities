import setka
import torch

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),  '..'))
import tiny_model
import test_dataset

from test_metrics import tensor_loss as loss

def test_ProgressBar():
    ds = test_dataset.CIFAR10()
    model = tiny_model.TensorNet()


    trainer = setka.base.Trainer(pipes=[
                                     setka.pipes.DatasetHandler(ds, batch_size=32, limits=2),
                                     setka.pipes.ModelHandler(model),
                                     setka.pipes.LossHandler(loss),
                                     setka.pipes.OneStepOptimizers(
                                        [
                                            setka.base.Optimizer(
                                                model,
                                                torch.optim.SGD,
                                                lr=0.1,
                                                momentum=0.9,
                                                weight_decay=5e-4)
                                        ]
                                     ),
                                     setka.pipes.ProgressBar()
                                 ])

    trainer.status['Test'] = ['This is just a very-very long string that helps in testing the package. This string has no delimiters so it will be split in the middle of the string']
    trainer.status['Test2'] = [1.0e-12, 1.0e12, 1, 100000000000000000, 1000]
    trainer.run_train(1)
