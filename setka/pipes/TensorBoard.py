from setka.pipes import Pipe

import torch.utils.tensorboard as TB
import sys
import os


class TensorBoard(Pipe):
    '''
    pipe to write the progress to the TensorBoard. When the epoch starts
    (before_epoch), it uploads computed metrics on previous epoch to the TensorBoard.

    It also writes the predictions to the TensorBoard when the ```predict```
    method of the Trainer is called and visualization function is specified.

    Visaulization function (passed as ```f``` to the construtor)
    takes as inputs: one input, target and output per sample and returns the
    dictionary with the outputs for visualization. This dictionary may contain
    the following keys:

    {
        "images": dict with numpy images,
        "texts": dict with texts,
        "audios": dict with numpy audios,
        "figures": dict with matplotlib figures,
        "graphs": dict with onnx graphs,
        "embeddings": dict with embeddings
    }
    Each of the dicts should have the following structure:
    {image_name: image} for images. For example, the following syntax will work:

    ```
    {"images": {"input": input_fig,
                "output": ouput_fig,
                "target": target_fig}}
    ```

    Args:
        f (callbale): function to visualize the network results.

        write_flag (bool): if True -- the results of f will be written to the tensorboard

        name (str): name of the experiment.

        log_dir (str): path to the directory for "tensorboard --logdir" command.

    '''
    def __init__(self,
                 f=None,
                 write_flag=True,
                 name='experiment_name',
                 log_dir='runs',
                 priority={'after_batch': 10}):
        self.f = f
        self.write_flag = write_flag
        self.log_dir = log_dir
        self.name = name

    def before_epoch(self):
        '''
        Writes scalars (metrics) of the previous epoch.
        '''
        if not self.write_flag:
            return None
        
        self.tb_writer = TB.SummaryWriter(log_dir=os.path.join(self.log_dir, self.name))
        
        if self.trainer._mode == 'train' and hasattr(self.trainer, '_metrics'):
            for subset in self.trainer._metrics:
                for metric_name in self.trainer._metrics[subset]:
                    self.tb_writer.add_scalar(
                        f'{metric_name}/{subset}',
                        self.trainer._metrics[subset][metric_name],
                        self.trainer._epoch - 1)

    def show(self, to_show, id):
        type_writers = {
            'images': self.tb_writer.add_image,
            'texts': self.tb_writer.add_text,
            'audios': self.tb_writer.add_audio,
            'figures': (lambda x, y, z: self.tb_writer.add_figure(x, y, z)),
            'graphs': self.tb_writer.add_graph,
            'embeddings': self.tb_writer.add_embedding}

        for type in type_writers:
            if type in to_show:
                for desc in to_show[type]:
                    type_writers[type](str(id) + '/' + desc,
                        to_show[type][desc], str(self.trainer._epoch))

    @staticmethod
    def get_one(input, item_index):
        if isinstance(input, (list, tuple)):
            one = []
            for list_index in range(len(input)):
                one.append(input[list_index][item_index])
            return one
        elif isinstance(input, dict):
            one = {}
            for dict_key in input:
                one[dict_key] = input[dict_key][item_index]
            return one
        else:
            one = input[item_index]
            return one

    def after_batch(self):
        '''
        Writes the figures to the tensorboard when the trainer is in the test mode.
        '''
        if not self.write_flag:
            return None
        
        if self.trainer._mode == 'test' and self.write_flag and (self.f is not None):
            for index in range(len(self.trainer._ids)):

                one_input = self.get_one(self.trainer._input, index)
                one_output = self.get_one(self.trainer._output, index)

                res = self.f(one_input, one_output)
                id = self.trainer._ids[index]

                self.show(res, id)

    def after_epoch(self):
        '''
        Destroys TensorBoardWriter
        '''
        if not self.write_flag:
            return None
        
        self.tb_writer.close()
        del self.tb_writer
