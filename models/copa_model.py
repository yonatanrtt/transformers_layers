import torch
import torch.nn as nn
import ipdb

class CopaModel(nn.Module):

    def __init__(self, _lm):
        super(CopaModel, self).__init__()
        self.lm = _lm        
        self.model = _lm.model
        n_class = 1
        self.linear = nn.Linear(_lm.config.hidden_size, n_class)
        self.act = nn.Sigmoid()
        self.dropout =  nn.Dropout(p=self.lm.config.hidden_dropout_prob)
        self.loss_fn = nn.CrossEntropyLoss()

    def lm_cls(self, _input):
      output = self.model(_input.input_ids, attention_mask=_input.attention_mask).last_hidden_state[:,0,:]
      return output

    def forward(self, _batch):
        # ipdb.set_trace()
        
        _positive_tokenized, _negative_tokenized, _labels = _batch

        cls_positive = self.lm_cls(_positive_tokenized)
        cls_negative = self.lm_cls(_negative_tokenized)

        cls_positive = cls_positive.view(cls_positive.shape[0], 1, cls_positive.shape[-1])
        cls_negative = cls_negative.view(cls_negative.shape[0], 1, cls_negative.shape[-1])

        embd = torch.cat((cls_positive, cls_negative), axis=1)
        
        output = self.dropout(embd)
        output = self.linear(output)
        output = output.view(output.shape[0], output.shape[1])

        loss = self.loss_fn(output, _labels.long())

        return output, loss, _labels

        