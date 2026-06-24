# Model Card: Nucleotide Transformer v2 50M, fine-tuned on GenomicBenchmarks

> **Note on availability:** The trained weight files are **not redistributed**. The base model is licensed CC-BY-NC-SA-4.0 (non-commercial, share-alike), and any derivative inherits those terms. This card documents the model that the published code reproduces. To obtain the checkpoints, run the training commands in the [repository README](README.md); every seed, split, and hyperparameter is fixed.

## Model description

Fine-tuned version of [InstaDeepAI/nucleotide-transformer-v2-50m-multi-species](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species) for binary DNA sequence classification on [GenomicBenchmarks](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks) tasks.

- **Developed by:** Ankur Sharma
- **Model type:** Genomic sequence classifier (full fine-tune of a foundation model)
- **Base model:** `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`
- **Architecture:** ESM-style transformer, 53.8M parameters, 6-mer tokenization
- **Fine-tuning method:** Full fine-tuning (all 53.8M params + 2-layer classification head)
- **Tasks trained on:** `human_enhancers_cohn`, `human_nontata_promoters`
- **License:** Code MIT; **fine-tuned weights CC-BY-NC-SA-4.0** (inherited from the base model; non-commercial, share-alike).

---

## Base model

`InstaDeepAI/nucleotide-transformer-v2-50m-multi-species` is a genomic foundation model pretrained by InstaDeep on multi-species genomes using an ESM-style masked-language-modelling objective with 6-mer tokenization (~50M parameters).

---

## Training data

**Task 1:** `human_enhancers_cohn`
- Binary: human enhancer sequences vs. background
- 20,843 train / 6,948 test, balanced 50/50
- Source: [katarinagresova/Genomic_Benchmarks_human_enhancers_cohn](https://huggingface.co/datasets/katarinagresova/Genomic_Benchmarks_human_enhancers_cohn)

**Task 2:** `human_nontata_promoters`
- Binary: non-TATA promoter sequences vs. background
- 27,097 train / 9,034 test
- Source: [katarinagresova/Genomic_Benchmarks_human_nontata_promoters](https://huggingface.co/datasets/katarinagresova/Genomic_Benchmarks_human_nontata_promoters)

---

## Training procedure

**Hardware:** Apple M4 Pro, 48 GB unified memory, PyTorch MPS backend (`PYTORCH_ENABLE_MPS_FALLBACK=1`).

**Hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Training set | Full task training split (shuffled, balanced) |
| Validation split | 15% of train held out for checkpoint selection |
| Reverse complement augmentation | enabled (doubles effective dataset) |
| Epochs | 3–4 (best checkpoint typically at epoch 1–2) |
| Batch size | 16 |
| Learning rate | 1e-5 (peak) |
| LR schedule | Linear warmup 300 steps → cosine decay to 0 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Gradient clipping | max_norm = 1.0 |
| Max sequence length | 500 tokens |
| Random seeds | 42, 0, 123 (enhancers); 42 (promoters) |
| Checkpoint selection | Best by validation MCC; test evaluated once at end |

---

## Evaluation results

Best checkpoint selected on a 15% validation holdout; the test split was evaluated **exactly once** at the end (no test-set leakage during training).

### Task 1: `human_enhancers_cohn` (6,948 test examples), 3 seeds (42, 0, 123)

| Model | Accuracy | F1 | MCC |
|---|---|---|---|
| Base (pretrained backbone, untrained head) | 0.499 | 0.021 | -0.009 |
| **Fine-tuned (this model)** | **0.735 ± 0.004** | **0.745 ± 0.016** | **0.478 ± 0.003** |
| GB-CNN (Gresova et al. 2023) | 0.69 | — | — |
| DNABERT (Ji et al. 2021) | 0.706 | — | — |
| HyenaDNA tiny-1k (Nguyen et al. 2023) | 0.74 | — | — |
| NT-v2 500M (Dalla-Torre et al. 2023) | 0.776 | — | — |
| DNABERT-2 (Zhou et al. 2023) | 0.785 | — | — |

### Task 2: `human_nontata_promoters` (9,034 test examples), single seed (42)

| Model | Accuracy | F1 | MCC |
|---|---|---|---|
| Base (pretrained backbone, untrained head) | 0.451 | 0.064 | -0.044 |
| **Fine-tuned (this model)** | **0.872** | **0.878** | **0.747** |
| GB-CNN (Gresova et al. 2023) | 0.85 | — | — |
| HyenaDNA tiny-1k (Nguyen et al. 2023) | 0.96 | — | — |

> Published values are taken from the source papers and were computed under their own conditions (different splits and preprocessing). They are reference points, not controlled head-to-head comparisons. MCC is reported because it is informative under class imbalance.

---

## Intended use

- Research and educational purposes.
- Demonstrating full fine-tuning of a genomic foundation model on benchmark tasks.
- A starting point for adapting NT-v2 to related DNA sequence classification problems.

## Limitations and out-of-scope use

- **Research only.** Not validated for clinical, diagnostic, or patient-facing use.
- Trained on single tasks; does not generalise to other organisms or tasks without retraining.
- The 50M-parameter variant has limited capacity; long-range genomic dependencies beyond its context window are out of scope.
- Results reflect one run on one hardware configuration (Apple M4 Pro, MPS). Numbers may vary with seed, hardware, or PyTorch version.
- `transformers==4.55.2` is pinned; v5.x removes `find_pruneable_heads_and_indices` which breaks the NT remote code.

---

## Related work

For context on the broader research area, see Fesser, Zhang, Li, Zitnik et al., *How Post-Training Shapes Biological Reasoning Models* ([arXiv:2606.16517](https://arxiv.org/abs/2606.16517), 2026), a cluster-scale study of continued-pretraining, supervised fine-tuning, and reinforcement-learning dynamics across DNA, RNA, and protein modalities. Its Finding 1, that supervised fine-tuning improves in-domain accuracy while out-of-domain performance peaks early and declines, has an in-domain counterpart in this work: validation MCC peaked at epoch 1–2 and then declined as the training loss continued to fall, which is why this pipeline uses a low learning rate and validation-based checkpoint selection rather than longer training. This model is a single controlled fine-tune, not a comparable research program; the connection is one of shared principle (clean evaluation, and the observation that more training is not always better), not scope.

---

## Citation

If you use this fine-tuning code, please cite the base model and benchmark:

**Nucleotide Transformer:**
```bibtex
@article{dallatorre2023nucleotide,
  title={The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics},
  author={Dalla-Torre, Hugo and Gonzalez, Liam and Mendoza-Revilla, Javier and others},
  journal={bioRxiv},
  year={2023}
}
```

**GenomicBenchmarks:**
```bibtex
@article{grevsova2023genomic,
  title={Genomic benchmarks: a collection of datasets for genomic sequence classification},
  author={Gre{\v{s}}ov{\'a}, Katar{\'i}na and Martinek, Vlastimil and {\v{C}}ech{\'a}k, David and others},
  journal={BMC Genomic Data},
  year={2023}
}
```

---

## License & attribution

The base model `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species` is licensed under **CC-BY-NC-SA-4.0**. These fine-tuned weights are a derivative and therefore inherit the same license:

- **Non-commercial use only.** Commercial use requires a separate license from InstaDeep.
- **ShareAlike.** Redistribution and derivatives must use CC-BY-NC-SA-4.0.
- **Attribution.** Credit InstaDeep (base model) and the GenomicBenchmarks authors (data).

The training/evaluation **code** in this repository is MIT-licensed.

This is a personal open-source project, developed independently in a personal capacity by Ankur Sharma. It is not affiliated with, endorsed by, or representative of any current or former employer, and uses only public models and public benchmark datasets. All views and results are the author's own.
