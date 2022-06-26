from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForMaskedLM


def init(_lm_name="roberta-base"):
    tokenizer = AutoTokenizer.from_pretrained(_lm_name)
    config = AutoConfig.from_pretrained(_lm_name)
    model = AutoModel.from_pretrained(_lm_name, config=config)
    model_mlm = AutoModelForMaskedLM.from_pretrained(_lm_name, config=config)
    return LM(tokenizer, config, model, model_mlm)

class LM():

  def __init__(self, _tokenizer, _config, _model, _model_mlm) -> None:
      self.tokenizer = _tokenizer
      self.config = _config
      self.model = _model
      self.model_mlm = _model_mlm
    