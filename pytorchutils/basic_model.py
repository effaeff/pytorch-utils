"""Basic neural network interface"""

import warnings
import copy
import abc
import misc

from pytorchutils.globals import nn

class BasicModel(nn.Module, metaclass=abc.ABCMeta):
    """Class for basic model"""
    def __init__(self, config):
        super().__init__()

        # Using a deepcopy of the configuration before using it,
        # to avoid any potential mutation when iterating asynchronously over configurations
        self.config = copy.deepcopy(config)

        # All models share some basics hyper parameters,
        # this is the section where they are copied into the model
        self.input_size = config.get('input_size', 0)
        self.output_size = config.get('output_size', 0)
        self.nb_units = config.get('nb_units', None)
        self.nb_layers = config.get('nb_layers', 0)
        if self.input_size == 0:
            warnings.warn("Warning: No input size defined.")
        if self.output_size == 0:
            raise ValueError("Error: No output size defined.")
        # if self.nb_units is None:
            # warnings.warn(
                # "Warning: No number of hidden units per layer defined. "
                # "This value has to be specified for most neural network architectures."
            # )
        if self.nb_layers == 0:
            warnings.warn(
                "Warning: No number of hidden layers defined."
            )
        # No dropout by default
        self.dropout_rate = self.config.get('dropout_rate', 0.0)

    @misc.lazy_property
    def dropout(self):
        """Define dropout layer"""
        return nn.Dropout(p=self.dropout_rate)

    @misc.lazy_property
    def activation(self):
        """Define activation function"""
        activation_label = self.config.get('activation', 'Sigmoid')
        return getattr(nn, activation_label)()

    @abc.abstractmethod
    def forward(self, __):
        """
        This function is usually common to all models ans should be overriden by the model.
        Returns:
            Prediction
            Regularization term
        """
        return
