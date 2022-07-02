import torch
import torch.nn as nn


class RteModel(nn.Module):

    def __init__(self, _lm):
        super(RteModel, self).__init__()
        self.lm = _lm
        self.model = _lm.model
        n_class = 2
        self.linear = nn.Linear(_lm.config.hidden_size, n_class)
        self.act = nn.Sigmoid()
        self.dropout =  nn.Dropout(p=self.lm.config.hidden_dropout_prob)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, _batch):
        _labels, _positive_tokenized, _negative_tokenized = _batch

        cls_positive = self.model(_positive_tokenized).last_hidden_state[:,0,:]
        cls_negative = self.model(_negative_tokenized).last_hidden_state[:,0,:]

        cls_positive = cls_positive.view(cls_positive.shape[0], 1, cls_positive.shape[-1])
        cls_negative = cls_negative.view(cls_negative.shape[0], 1, cls_negative.shape[-1])

        embd = torch.cat((cls_positive, cls_negative), axis=1)
        
        output = self.dropout(embd)
        output = self.linear(output)
        output = self.act(output)

        loss = self.loss_fn(output, _labels)

        return output, loss, loss

        