from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForMaskedLM


def init(_lm_name="roberta-base"):
    tokenizer = AutoTokenizer.from_pretrained(_lm_name)
    config = AutoConfig.from_pretrained(_lm_name)
    model = AutoModel.from_pretrained(_lm_name, config=config)
    model_mlm = AutoModelForMaskedLM.from_pretrained(_lm_name, config=config)
    return tokenizer, config, model, model_mlm
