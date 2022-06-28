from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForMaskedLM

class LM():

  def __init__(self, _lm_name="roberta-base", _is_pretrain=False) -> None:
      self.tokenizer = AutoTokenizer.from_pretrained(_lm_name)
      self.config = AutoConfig.from_pretrained(_lm_name)

      if _is_pretrain:
        self.config.output_hidden_states = True
        self.model_mlm = AutoModelForMaskedLM.from_pretrained(_lm_name, config=config)
      else:
        self.model = AutoModel.from_pretrained(_lm_name, config=self.config)
    