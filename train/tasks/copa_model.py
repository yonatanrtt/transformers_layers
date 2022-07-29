import torch
import torch.nn as nn

class CopaModel(nn.Module):

    def __init__(self, _lm):
        super(CopaModel, self).__init__()
        self.lm = _lm
        self.model = _lm.model_mlm 
        n_class = 1 
        self.linear = nn.Linear(_lm.config.hidden_size, n_class) 
        self.act = nn.Sigmoid() 
        self.dropout = nn. Dropout(p=self.lm.config.hidden_dropout_prob) 
        self.loss_fn = nn.CrossEntropyLoss()
    
    def lm_cls(self, _input):
        output = self.model(_input.input_ids, attention_mask=_input.attention_mask).hidden_states[-1][:, 0, :]
        return output
    
    def forward(self, _batch):
        _input_tokenized, _input_2_tokenized, _input_masked, _masked_label, _labels = _batch
        cls_input_1 = self.lm_cls(_input_tokenized) 
        cls_input_2 = self.lm_cls(_input_2_tokenized)
        cls_input_1 = cls_input_1.view(cls_input_1. shape[0], 1, cls_input_1.shape[-1])
        cls_input_2 = cls_input_2.view(cls_input_2. shape[0], 1, cls_input_2. shape[-1])
        embed = torch.cat((cls_input_1, cls_input_2), axis=1)
        output = self.dropout(embed) 
        output = self.linear(output) 
        output = output.view(output.shape[0], output.shape[1])
        loss = self.loss_fn(output, _labels.long()) 
        return output, loss, _labels
