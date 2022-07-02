import torch.nn as nn

class Generator(nn.Module):

    def __init__(self, _lm, _noise_size=100):
        super(Generator, self).__init__()
        self.output_size = _lm.config.hidden_size
        self.hidden_sizes = [_lm.config.hidden_size]
        self.dropout_rate=0.1
        layers = []
        hidden_sizes = [_noise_size] + self.hidden_sizes
        for i in range(len(hidden_sizes)-1):
            layers.extend([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(self.dropout_rate)])

        layers.append(nn.Linear(hidden_sizes[-1], self.output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, noise):
        output_rep = self.layers(noise)
        return output_rep
