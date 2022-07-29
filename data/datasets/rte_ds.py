import torch 

class RteDataset(torch.utils.data. Dataset):
    
    def __init__(self, _data, _lm):
        self.premise = list(_data["premise"]) 
        self.hypothesis = list(_data["hypothesis"])
        self.label = list(_data["label"]) if "label" in _data.keys() else [0] * len(self.premise)
        self.lm = _lm

    def __getitem__(self, _idx):
            premise = self.premise[_idx] 
            hypothesis = self.hypothesis[_idx]
            inputs = premise + self.lm.tokenizer.sep_token + hypothesis 
            label = self.label[_idx]
            return inputs, label
            
    def __len__(self):
        return len(self.label)

    def preprocess_batch(self, _batch, _data_collator, _device):
        inputs, label = list(zip(*_batch))
        inputs_tokenized = self.lm.tokenizer(list(inputs), padding=True, truncation=True, return_tensors="pt")
        batch = inputs_tokenized.to(_device),
        if self.lm.is_mlm:
            masked_input, masked_label = _data_collator(tuple(inputs_tokenized.input_ids.cpu())).values()
            batch += masked_input.to(_device), 
            batch += masked_label.to(_device),
        labels = torch.Tensor(label).to(_device) 
        batch += labels,
        return batch
