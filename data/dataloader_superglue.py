from transformers import DataCollatorForLanguageModeling 
from functools import partial 
from torch.utils.data import DataLoader 
import numpy as np 
import shared.utils as utils 
import shared.constants as constants 
import data.utils_datasets as utils_datasets

class DataSuperglue:

    def __init__(self, _lm, _data_name, _dataset_class, _data_limit=None): 
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=_lm.tokenizer, mlm=True, mlm_probability=0.15) 
        self.lm = _lm 
        self.data_limit = _data_limit 
        self.device = utils.get_device() 
        self.data_name = _data_name 
        self.dataset_class = _dataset_class 
        self.data = utils_datasets.get_data(_data_name)
    
    def get_db_dataloader(self): 
        return (
            self.get_dataloader(self.data["train"], True),
            self.get_dataloader(self.data["validation"], False), 
            self.get_dataloader(self.data["test"], False)
        )
    
    def get_dataloader(self, _data, _is_train):
        data = _data.sample(frac=1, random_state=1).reset_index(drop=True) 
        if _is_train and self.data_limit is not None:
            print(f"limited data: {self.data_limit}")
            data = data[:self.data_limit] 
        # data = data[:8] 
        ds = self.dataset_class(data, self.lm) 
        return DataLoader(ds, batch_size=constants.BATCH_SIZE, shuffle=True,
        collate_fn=partial(ds.preprocess_batch, _data_collator=self.data_collator,
        _device=self.device), worker_init_fn=np.random.seed(constants.SEED))

