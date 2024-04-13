import keras
from torch.utils.tensorboard import SummaryWriter
import torch

class TensorboardCallback(keras.callbacks.Callback):

    def __init__(self, log_dir=None):
        '''
        Init callback
        Args:
            log_dir : log directory
        '''
        self.writer = SummaryWriter(log_dir=log_dir)

    def on_epoch_end(self, epoch, logs=None):
        '''
        Record logs at epoch end
        '''

        # ---- Records all metrics (very simply)
        #
        # for k,v in logs.items():
        #     self.writer.add_scalar(k,v, epoch)

        # ---- Records and group specific metrics
        #
        self.writer.add_scalars('Accuracy',
                                {'Train': logs['accuracy'],
                                 'Validation': logs['val_accuracy']},
                                epoch)

        self.writer.add_scalars('Loss',
                                {'Train': logs['loss'],
                                 'Validation': logs['val_loss']},
                                epoch)
        # dummy_input = (torch.zeros(24, 24, 1), torch.zeros(24, 24, 1))
        # self.writer.add_graph(self.model, dummy_input )
