import pandas as pd
import fortepyan as ff
from omegaconf import OmegaConf

from comparison.data.tokenizer import QuantizedMidiEncoder
from comparison.data_new.dataset import MidiQuantizer as MQ
from comparison.data_new.dataset import MyTokenizedMidiDataset
from comparison.data_new.dataset import load_cache_dataset as lcd
from comparison.data_new.midiencoder import VelocityEncoder as VE
from comparison.data.dataset import MidiQuantizer as MQ_translation
from comparison.data.tokenizer import VelocityEncoder as VE_translation
from comparison.data.dataset import load_cache_dataset as lcd_translation
from comparison.data.dataset import MyTokenizedMidiDataset as MTD_translation


def main():
    dataset_name = "roszcz/maestro-v1-sustain"
    dataset_cfg = {
        "sequence_len": 128,
        "sequence_step": 42,
        "quantization": {
            "duration": 3,
            "velocity": 3,
            # 650 start bins sound nice :)
            "dstart": 3,
        },
    }
    cfg = OmegaConf.create(dataset_cfg)
    dataset = lcd(cfg, dataset_name, split="test")
    dataset_translation = lcd_translation(cfg, dataset_name, split="test")

    quantizer = MQ(
        n_duration_bins=cfg.quantization.duration,
        n_velocity_bins=cfg.quantization.velocity,
        n_dstart_bins=cfg.quantization.dstart,
    )

    q_translation = MQ_translation(
        n_duration_bins=cfg.quantization.duration,
        n_velocity_bins=cfg.quantization.velocity,
        n_dstart_bins=cfg.quantization.dstart,
    )

    record_translation = pd.DataFrame(dataset_translation[90])
    record = pd.DataFrame(dataset[90])
    print(record)
    print(record_translation)
    record = quantizer.apply_quantization(record)
    record_translation = q_translation.apply_quantization(record_translation)

    print(record.compare(record_translation))

    piece = ff.MidiPiece(record)
    piece_translation = ff.MidiPiece(record_translation)

    print(piece.df)
    print(piece_translation.df)

    # ff.view.make_piano_roll_video(piece, "test.mp4")

    # this is for testing and debugging btw
    encoder = VE(cfg.quantization, time_quantization_method="dstart")
    test_dataset = MyTokenizedMidiDataset(
        encoder=encoder,
        dataset=dataset,
        dataset_cfg=cfg,
    )
    src_encoder = QuantizedMidiEncoder(cfg.quantization)
    tgt_encoder = VE_translation()
    translation_dataset = MTD_translation(
        dataset=dataset_translation,
        dataset_cfg=cfg,
        src_encoder=src_encoder,
        tgt_encoder=tgt_encoder,
    )
    record = test_dataset[90]
    record_translation = translation_dataset[90]
    print(record["source_token_ids"])
    print(record["target_token_ids"])
    print(record_translation["source_token_ids"])
    print(record_translation["target_token_ids"])

    print([encoder.vocab[token] for token in record["source_token_ids"]])
    print([encoder.vocab[token] for token in record["target_token_ids"]])
    print([src_encoder.vocab[token] for token in record_translation["source_token_ids"]])
    print([tgt_encoder.vocab[token] for token in record_translation["target_token_ids"]])


if __name__ == "__main__":
    main()
