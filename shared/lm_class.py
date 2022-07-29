
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForMaskedLM 
import shared.constants as constants

class LM:

    def __init__(self, _lm_path=None, _is_pretrain=False) -> None: 
        if _lm_path is None:
            _lm_path = constants.LM_DEFAULT 
        self.tokenizer = AutoTokenizer.from_pretrained(_lm_path) 
        self.config = AutoConfig.from_pretrained(_lm_path) 
        self.lm_path = _lm_path
        if _is_pretrain:
            self.is_mlm = True 
            self.config.output_hidden_states = True
            self.model_mlm = AutoModelForMaskedLM.from_pretrained(_lm_path, config=self.config)
        else:
            self.is_mlm = False 
            self.model = AutoModel.from_pretrained(_lm_path, config=self.config, add_pooling_layer=False)
