import setka
import torch

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),  '..'))
import tiny_model
import test_dataset

import matplotlib.pyplot as plt

from test_metrics import dict_loss, list_loss, tensor_loss
from test_metrics import dict_acc, list_acc, tensor_acc

def test_TensorBoard():
    ds = test_dataset.CIFAR10()
    model = tiny_model.DictNet()

    def view_result(one_input, one_output):
        # print("In view result")
        img = one_input[0]
        img = (img - img.min()) / (img.max() - img.min() + 1.0e-4)
        truth = one_input[1]
        label = one_output['res']

        # print(img.size())

        fig = plt.figure()
        plt.imshow(img.permute(2, 1, 0))
        plt.close()
        return {'figures': {'img': fig}}

    trainer = setka.base.Trainer(pipes=[
                                     setka.pipes.DatasetHandler(ds, batch_size=4, limits=2),
                                     setka.pipes.ModelHandler(model),
                                     setka.pipes.LossHandler(dict_loss),
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
                                     setka.pipes.ComputeMetrics([dict_loss, dict_acc]),
                                     setka.pipes.TensorBoard(f=view_result)
                                 ])

    trainer.run_train(2)
    trainer.run_epoch(mode='test', subset='valid', n_iterations=2)


