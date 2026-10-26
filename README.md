# jlens — Jacobian lens

A fork of the [reference implementation](https://github.com/anthropics/jacobian-lens) for
the [Jacobian lens](https://transformer-circuits.pub/2026/workspace/index.html), modified to test
the sensitivity of the lens to the fitting corpus. **Currently a work in progress.**

## Question

How does the corpus used to fit a Jacobian lens affect the resulting lens? Qualitatively, are there differences in the
readouts from different lenses?

## Process

All lenses will be fitted on 1000 prompts from the dataset.

- [x] Adapt code to work on my laptop (Apple Silicon Mac)
- [ ] Run a proof-of-concept experiment locally with Qwen3.5-0.8B
    - [ ] Fit multiple lenses using samples of the same dataset (wikitext-103-raw-v1). Create a baseline for sampling
      variability.
    - [ ] Pick other datasets and fit multiple lenses on each
        * Refusals only
        * Math (proofs)
        * Non-English. Using Chinese would be interesting since Qwen and a lot of other open models already have Chinese
          characters in their readouts
        * ASCII art
        * Encodings (e.g. base64)
        * Poetry
        * Gibberish
        * Code that crashes (does this influence the "ValueError" and similar readouts?)
    - [ ] Calculate Frobenius distance, cosine similarity, other matrix metrics
    - [ ] Calculate top-k overlap, KL divergence between lenses
    - [ ] Find qualitative examples of different readouts from the different lenses, if they exist
        * Start by checking the provided [examples](/data/evaluations)
        * Consider vibe-coding a UI for lens readouts, similar to Neuronpedia
- [ ] Run a full experiment on a larger model using rented GPUs

## Why?

If the J-lens is not robust to the fitting prompt distribution, more care will need to be taken when fitting the lens.

Also, I've been learning about mech interp and wanted to try some hands-on experimentation. :)

## Run the experiment

From the repository root:

```bash
uv run --extra experiment corpus-experiment
```

Configure the model, dataset, and fitting settings in [corpus_experiment/main.py](corpus_experiment/main.py).
