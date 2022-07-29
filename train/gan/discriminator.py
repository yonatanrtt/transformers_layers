import torch.nn as nn 
import torch

class Discriminator(nn. Module):

    def __init__(self, _lm):
        super(Discriminator, self).__init__()
        hidden_size = _lm.config.hidden_size
        num_hidden_layers = _lm.config.num_hidden_layers + 1 # + embedding layer
        dropout_rate = _lm.config.hidden_dropout_prob 
        n_layers = num_hidden_layers + 1 # + 1 for not real - generated 
        self.layers = nn.Sequential(
            nn.Linear(_lm.config.hidden_size, hidden_size), 
            nn.LeakyReLU(), 
            nn. Dropout(dropout_rate),
        )
        self.linear = nn.Linear(hidden_size, n_layers) 
        self.softmax = nn.Softmax(dim=1)
        print(f"n_layers: {n_layers}")
        
    def forward(self, _input_data):
        _inputs, _labels = zip(*_input_data) 
        inputs = torch.stack(_inputs) 
        output = self.layers(inputs) 
        logit = self.linear(output) 
        prob = self.softmax(logit) 
        return output, logit, prob, _labels
