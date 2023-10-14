from omegaconf import OmegaConf
from transformers import T5Config, T5ForConditionalGeneration

from data.tokenizer import MultiTokEncoder
from data.dataset import MyTokenizedMidiDataset, load_cache_dataset

cfg = {
    "sequence_len": 128,
    "sequence_step": 10,
    "quantization": {
        "duration": 3,
        "velocity": 3,
        "dstart": 3,
    },
}
cfg = OmegaConf.create(cfg)

tokenizer = MultiTokEncoder(cfg.quantization)
config = T5Config(
    vocab_size=tokenizer.vocab_size,
    decoder_start_token_id=0,
    use_cache=False,
    output_attentions=False,
    output_scores=False,
    output_hidden_states=False,
    return_dict=False,
)
print(tokenizer.vocab_size)
model = T5ForConditionalGeneration(config)

print([len(model.state_dict()[key]) for key in model.state_dict().keys()])

dataset = load_cache_dataset(cfg, "roszcz/maestro-v1-sustain", split="test")


train_dataset = MyTokenizedMidiDataset(
    dataset=dataset,
    dataset_cfg=cfg,
    encoder=tokenizer,
)

record = train_dataset[0]
print(record)

outputs = model(input_ids=record["source_token_ids"].unsqueeze(0), labels=record["target_token_ids"].unsqueeze(0))
print(outputs)
