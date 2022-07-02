import torch
import torch.nn as nn


class ModelMiddleTrained(nn.Module):

    def __init__(self, _lm):
        super(ModelMiddleTrained, self).__init__()
        self.lm = _lm
        self.model = _lm.model
        self.layers_linear = nn.Linear(_lm.config.hidden_size, _lm.config.num_hidden_layers)
        self.act = nn.Sigmoid()
        self.dropout =  nn.Dropout(p=self.lm.config.hidden_dropout_prob)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, _input_tokenized, _input_masked):

        # todo - orthogonal matrices + gan
        output_lm = self.model(_input_tokenized)

        layers_classifaiers = []
        for layer in output_lm.hidden_states:
          cls = layer[:, 0, :]
          output = self.dropout(cls)
          output = self.layers_linear(output)          
          layers_classifaiers.append(output)
        layers_loss_sum = sum(layers_classifaiers)

        output_mlm = self.model(_input_masked, labels=_input_tokenized)
        output_mlm.loss


        return output

    def save_lm(self):
      self.lm.tokenizer.save_pretrained("")
      self.model.save_pretrained()

        