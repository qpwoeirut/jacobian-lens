import logging
import os

import torch
import transformers

import jlens
from jlens.examples import load_wikitext_prompts

jlens.configure_logging()
logger = logging.getLogger("jlens.scratch")
logger.info("Starting fitting process")

MODEL_NAME = "Qwen/Qwen3.5-0.8B"

# Concurrent weight copy/cast operations can crash PyTorch's MPS shader cache.
os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"

hf_model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map={"": "mps"},
)
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = jlens.from_hf(hf_model, tokenizer)
logger.info("Loaded model")

prompts = load_wikitext_prompts(n_prompts=1000)
logger.info("Loaded prompts")

lens = jlens.fit(
    model,
    prompts,
    dim_batch=32,
    max_seq_len=128,
    checkpoint_path=f"ckpt-{MODEL_NAME.rsplit('/', 1)[-1]}.pt",
)
lens.save("jacobian_lens.pt")

logger.info("Done")
