import torch
import torch.nn as nn


class CopaModel(nn.Module):

    def __init__(self, _lm, _device):
        super(CopaModel, self).__init__()
        self.model = _lm.model
        self.linear = nn.Linear(_lm.config.hidden_size, 1).to(_device)
        self.act = nn.Sigmoid()

    def forward(self, _positive_tokenized, _negative_tokenized, positive_input, negative_input):

        cls_positive = self.model(_positive_tokenized).last_hidden_state [:,0,:]
        cls_negative = self.model(_negative_tokenized).last_hidden_state [:,0,:]

        cls_positive = cls_positive.view(cls_positive.shape[0], 1, cls_positive.shape[-1])
        cls_negative = cls_negative.view(cls_negative.shape[0], 1, cls_negative.shape[-1])

        embd = torch.cat((cls_positive, cls_negative), axis=1)
        
        linear = self.linear(embd)
        output = self.act(linear)

        return output

        