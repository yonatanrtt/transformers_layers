import torch
import torch.nn as nn


class CopaModel(nn.Module):

    def __init__(self, _lm, _device):
        super(CopaModel, self).__init__()
        self.model = _lm.model
        self.layers_linear = nn.Linear(_lm.config.hidden_size, _lm.config.num_hidden_layers).to(_device)

    def forward(self, _positive_tokenized, _negative_tokenized, positive_input, negative_input):

        # todo - orthogonal matrices + gan
        positive = self.model(_positive_tokenized)
        negative = self.model(_negative_tokenized)

        layers_classifaiers = []
        for layer in positive.hidden_states:
          cls = layer[:, 0, :]
          output = self.layers_linear(cls)
          layers_classifaiers.append(output)

        for layer in negative.hidden_states:
          cls = layer[:, 0, :]
          output = self.layers_linear(cls)
          layers_classifaiers.append(output)

        positive_mask = model(input_ids, attention_mask=input_mask, labels=token_labels)
        negative_mask = model(input_ids, attention_mask=input_mask, labels=token_labels)


        return output

        