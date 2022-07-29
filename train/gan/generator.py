import torch.nn as nn 
import torch 
import shared.utils as utils 
import shared.constants as constants

class Generator(nn. Module):

    def __init__(self, _lm):
        super(Generator, self). __init__()
        self.hidden_size = _lm.config.hidden_size 
        self.dropout_rate = _lm.config.hidden_dropout_prob 
        self.device = utils.get_device() 
        self.layers = nn. Sequential(
            nn.Linear(self.hidden_size, self.hidden_size), 
            nn.LeakyReLU(), 
            nn.Dropout(self.dropout_rate),
        )
        self.linear = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self):
        inputs = torch.rand((constants.BATCH_SIZE, self.hidden_size)).to(self.device)
        # GAN-BERT - uses loop here for self.layers (inputs) 
        output = self.layers(inputs) 
        # separate from the loop 
        output = self.linear(output) 
        output_list = list(output) 
        output = list(zip(output_list, [0] * len(output_list))) 
        return output
