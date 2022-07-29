import torch 

class CopaDataset(torch.utils.data.Dataset):

    def __init__(self, _data, _lm):
        self.premise = list(_data["premise"]) 
        self.choice1 = list(_data["choice1"])
        self.choice2 = list(_data["choice2"]) 
        self.question = list(_data["question"])
        self.label = list(_data["label"]) if "label" in _data.keys() else [0] * len(self.premise)
        self.lm = _lm

    def __getitem__(self, _idx):
        connector = self.lm.tokenizer.sep_token 
        answers = [self.choice1[_idx], self.choice2[_idx]]
        label = self.label[_idx]
        first_attr = [self.premise[_idx]] * 2 
        second_attr = answers
        if self.question[_idx] == "cause":
            first_attr, second_attr = second_attr, first_attr            
        input_1 = first_attr[0] + connector + second_attr[0] 
        input_2 = first_attr[1] + connector + second_attr[1]
        positive = input_1 if label == 0 else input_2 
        return input_1, input_2, positive, label

    def __len__(self):
        return len(self.label) 
    
    def preprocess_batch(self, _batch, _data_collator, _device):
        input_1, input_2, positive, label = list(zip(*_batch))
        input_1_tokenized = self.lm.tokenizer(list(input_1), padding=True, truncation=True, return_tensors="pt")
        input_2_tokenized = self.lm.tokenizer(list(input_2), padding=True, truncation=True, return_tensors="pt")
        batch = (input_1_tokenized.to(_device), input_2_tokenized.to(_device))

        if self.lm.is_mlm:
            positive_tokenized = self.lm.tokenizer(list(positive), padding=True, truncation=True, return_tensors="pt")
            masked_input, masked_label = _data_collator(tuple(positive_tokenized.input_ids.cpu())).values()
            batch += masked_input.to(_device), 
            batch += masked_label.to(_device),
        labels = torch. Tensor(label).to(_device) 
        batch += labels,
        return batch

