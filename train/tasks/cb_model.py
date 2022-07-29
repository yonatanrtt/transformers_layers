import torch.nn as nn

class CbModel(nn.Module):

    def __init__(self, _lm):
        super(CbModel, self).__init__()
        self.lm = _lm 
        self.model = _lm.model_mlm 
        n_class = 3 
        self.linear = nn.Linear(_lm.config.hidden_size, n_class) 
        self.act = nn.Sigmoid() 
        self.dropout = nn.Dropout(p=self.lm.config.hidden_dropout_prob) 
        self.loss_fn = nn.CrossEntropyLoss()
   
    def forward(self, _batch):
        _input_tokenized, input_masked, _masked_label, _labels = _batch 
        cls_inputs = self.model(_input_tokenized.input_ids, attention_mask=_input_tokenized.attention_mask).hidden_states[-1][:, 0, :]
        output = self.dropout(cls_inputs) 
        output = self.linear(output) 
        loss = self.loss_fn(output, _labels.long())
        return output, loss, _labels
