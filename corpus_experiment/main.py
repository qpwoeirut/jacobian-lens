import logging
import os
import random

import torch
import transformers
from datasets import load_dataset

import jlens
from corpus_experiment import logs

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
DATASET_PATH = "Salesforce/wikitext"
DATASET_NAME = "wikitext-103-raw-v1"

TRIALS = 5
N_PROMPTS = 1000
MIN_PROMPT_CHARS = 600
MAX_SEQ_TOKENS = 128


def load_prompts(path: str, name: str, *, min_chars: int) -> list[str]:
    """Return all records of at least min_chars characters."""
    dataset = load_dataset(path=path, name=name, split="train", streaming=False)
    return [record["text"] for record in dataset if len(record["text"]) >= min_chars]


def main():
    logs.configure_logging()

    # Concurrent weight copy/cast operations can crash PyTorch's MPS shader cache.
    os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
    prompts = load_prompts(DATASET_PATH, DATASET_NAME, min_chars=MIN_PROMPT_CHARS)
    logger.info("Loaded %d prompts", len(prompts))

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map={"": "mps"},
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = jlens.from_hf(hf_model, tokenizer)
    logger.info("Loaded model")

    for trial in range(TRIALS):
        logger.info("Starting fitting process for trial %d/%d", trial + 1, TRIALS)
        random.seed(trial)
        sampled_prompts = random.choices(prompts, k=N_PROMPTS)

        model_shortname = MODEL_NAME.rsplit('/', 1)[-1]
        lens = jlens.fit(
            model,
            sampled_prompts,
            dim_batch=32,
            max_seq_len=MAX_SEQ_TOKENS,
            checkpoint_path=f"ckpt_{model_shortname}_{DATASET_NAME}_{trial}.pt",
            checkpoint_every=10,
        )
        lens.save(f"lens_{model_shortname}_{DATASET_NAME}_{trial}.pt")

    logger.info("Done")


if __name__ == '__main__':
    main()
