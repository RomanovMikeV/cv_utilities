from setka.pipes.Pipe import Pipe
from apex import amp

class UseApex(Pipe):
    """
    This pipe forces the model to be trained in fp16 mode provided
    by apex library.
    """

    def __init__(self):
        super(UseApex, self).__init__()
        self.set_priority({'on_init': 10})

    def on_init(self):
        self.trainer._fp16 = True