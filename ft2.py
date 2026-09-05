import gc
import os
import pandas as pd
import torch
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.models.xtts import XttsAudioConfig

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

MODEL_DIR    = r"D:\TTS\models\models--coqui--XTTS-v2\snapshots\6c2b0d75eae4b7047358e3b6bd9325f857d43f77"
DATASET_PATH = r"D:\TTS\ljspeech_output"
OUTPUT_DIR   = r"D:\TTS\finetuned"
SPEAKER_NAME = "spek1"
LANGUAGE     = "hi"

BATCH_SIZE    = 1        # 6GB safe
GRAD_ACCUM    = 8        # effective batch = 8
NUM_EPOCHS    = 10
MAX_AUDIO_LEN = 220500   # ~10s max audio length
EVAL_SPLIT    = 0.15


# ─────────────────────────────────────────────
# DATASET CONVERSION
# ─────────────────────────────────────────────

def convert_ljspeech_to_coqui(dataset_path, speaker_name, eval_split=0.15):
    src       = os.path.join(dataset_path, "metadata.csv")
    train_out = os.path.join(dataset_path, "metadata_train.csv")
    eval_out  = os.path.join(dataset_path, "metadata_eval.csv")

    if os.path.isfile(train_out) and os.path.isfile(eval_out):
        print(" > Coqui CSVs already exist, skipping conversion.")
        return train_out, eval_out

    rows = []
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")

            if len(parts) < 2:
                continue

            filename = parts[0].strip().replace("\\", "/")

            # Text column
            if len(parts) >= 3:
                text = parts[2].strip()
            else:
                text = parts[1].strip()

            # Remove "wavs/" if already present
            if filename.startswith("wavs/"):
                filename = filename[5:]

            # Remove ".wav" if already present
            if filename.lower().endswith(".wav"):
                filename = filename[:-4]

            wav_rel = f"wavs/{filename}.wav"

            rows.append({
                "audio_file": wav_rel,
                "text": text,
                "speaker_name": speaker_name,
            })

    df = pd.DataFrame(rows).sample(frac=1, random_state=42)
    n_eval = max(1, int(len(df) * eval_split))

    df[:n_eval].sort_values("audio_file").to_csv(eval_out,  sep="|", index=False)
    df[n_eval:].sort_values("audio_file").to_csv(train_out, sep="|", index=False)

    print(f" > Converted {len(df) - n_eval} train / {n_eval} eval samples")
    return "metadata_train.csv", "metadata_eval.csv"


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # VRAM optimizations
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    torch.cuda.empty_cache()

    train_csv, eval_csv = convert_ljspeech_to_coqui(DATASET_PATH, SPEAKER_NAME, EVAL_SPLIT)

    XTTS_CHECKPOINT = os.path.join(MODEL_DIR, "model.pth")
    TOKENIZER_FILE  = os.path.join(MODEL_DIR, "vocab.json")
    DVAE_CHECKPOINT = os.path.join(MODEL_DIR, "dvae.pth")
    MEL_NORM_FILE   = os.path.join(MODEL_DIR, "mel_stats.pth")

    for label, path in [
        ("XTTS checkpoint", XTTS_CHECKPOINT),
        ("Tokenizer/vocab",  TOKENIZER_FILE),
        ("DVAE checkpoint",  DVAE_CHECKPOINT),
        ("Mel stats",        MEL_NORM_FILE),
    ]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f" [!] {label} not found at: {path}")
        print(f" > Found {label}: {path}")

    config_dataset = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="ft_dataset",
        path=DATASET_PATH,
        meta_file_train="metadata_train.csv",
        meta_file_val="metadata_eval.csv",
        language=LANGUAGE,
    )

    model_args = GPTArgs(
        max_conditioning_length=66150,   # 3s reference audio
        min_conditioning_length=22050,   # 1s minimum
        debug_loading_failures=False,
        max_wav_length=MAX_AUDIO_LEN,    # 6s max training audio
        max_text_length=200,
        mel_norm_file=MEL_NORM_FILE,
        dvae_checkpoint=DVAE_CHECKPOINT,
        xtts_checkpoint=XTTS_CHECKPOINT,
        tokenizer_file=TOKENIZER_FILE,
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    audio_config = XttsAudioConfig(
        sample_rate=22050,
        dvae_sample_rate=22050,
        output_sample_rate=24000,
    )

    OUT_PATH = os.path.join(OUTPUT_DIR, "run", "training")

    config = GPTTrainerConfig(
        epochs=NUM_EPOCHS,
        output_path=OUT_PATH,
        model_args=model_args,
        run_name="GPT_XTTS_FT",
        project_name="XTTS_trainer",
        run_description="XTTS v2 fine-tuning — new speaker",
        dashboard_logger="tensorboard",
        logger_uri=None,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=8,          # reduced from 48
        eval_batch_size=1,
        num_loader_workers=2,        # reduced from 4
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        log_model_step=100,
        save_step=1000,
        save_n_checkpoints=1,
        save_checkpoints=True,
        print_eval=False,
        mixed_precision=True,        # fp16 — halves VRAM usage
        optimizer="AdamW",
        optimizer_wd_only_on_weights=True,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000, 150000, 300000], "gamma": 0.5, "last_epoch": -1},
        test_sentences=[],
    )

    model = GPTTrainer.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(
        [config_dataset],
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    print(f" > Train samples : {len(train_samples)}")
    print(f" > Eval  samples : {len(eval_samples)}")

    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=False,
            grad_accum_steps=GRAD_ACCUM,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    trainer.fit()

    samples_len = [len(item["text"].split(" ")) for item in train_samples]
    speaker_ref = train_samples[samples_len.index(max(samples_len))]["audio_file"]

    print(f"\n > Training complete!")
    print(f" > Best model saved to  : {trainer.output_path}")
    print(f" > Suggested speaker ref: {speaker_ref}")

    del model, trainer, train_samples, eval_samples
    gc.collect()
    torch.cuda.empty_cache()