"""
Multilayer Perceptron model using PyTorch.
"""

import misc
from pytorchutils.basic_model import BasicModel
from pytorchutils.globals import nn

class MLPModel(BasicModel):
    """
    Actual class for MLP model.
    """
    def __init__(self, config):
        BasicModel.__init__(self, config)
        self.input_layer
        self.hidden_layer
        self.output_layer

    @misc.lazy_property
    def hidden_layer(self):
        """Property for set of hidden layers"""
        hidden_layer = nn.ModuleList()
        for __ in range(self.nb_layers):
            layer = nn.Linear(self.nb_units, self.nb_units)
            self.init_weights(layer)
            hidden_layer.append(layer)
        return hidden_layer

    @misc.lazy_property
    def input_layer(self):
        """Property for input layer"""
        layer = nn.Linear(self.input_size, self.nb_units)
        self.init_weights(layer)
        return layer

    @misc.lazy_property
    def output_layer(self):
        """Property for output layer"""
        layer = nn.Linear(self.nb_units, self.output_size)
        self.init_weights(layer)
        return layer

    def init_weights(self, layer):
        """Initialize weights of layer"""
        init_label = self.config.get('init', 'xavier_normal') # Xavier normal being default
        for name, param in layer.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                getattr(nn.init, '{}_'.format(init_label))(param)

    def forward(self, inp):
        """Forwars pass through MLP layers"""
        pred_out = self.input_layer(inp)
        pred_out = self.activation(pred_out)
        pred_out = self.dropout(pred_out)
        for layer in self.hidden_layer:
            pred_out = layer(pred_out)
            pred_out = self.activation(pred_out)
            pred_out = self.dropout(pred_out)
        pred_out = self.output_layer(pred_out)

        return pred_out
