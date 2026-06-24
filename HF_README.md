---
license: cc-by-nc-sa-4.0
base_model: InstaDeepAI/nucleotide-transformer-v2-50m-multi-species
tags:
  - genomics
  - dna
  - sequence-classification
  - nucleotide-transformer
  - fine-tuning
  - bioinformatics
datasets:
  - katarinagresova/Genomic_Benchmarks_human_enhancers_cohn
  - katarinagresova/Genomic_Benchmarks_human_nontata_promoters
language:
  - en
library_name: transformers
pipeline_tag: text-classification
metrics:
  - matthews_correlation
  - f1
  - accuracy
model-index:
  - name: nucleotide-transformer-v2-50m-genomicbenchmarks-ft
    results:
      - task:
          type: text-classification
          name: DNA sequence classification (enhancers)
        dataset:
          name: GenomicBenchmarks human_enhancers_cohn
          type: katarinagresova/Genomic_Benchmarks_human_enhancers_cohn
        metrics:
          - type: matthews_correlation
            value: 0.478
            name: Test MCC (mean of 3 seeds)
          - type: accuracy
            value: 0.735
            name: Test accuracy (mean of 3 seeds)
          - type: f1
            value: 0.745
            name: Test F1 (mean of 3 seeds)
      - task:
          type: text-classification
          name: DNA sequence classification (promoters)
        dataset:
          name: GenomicBenchmarks human_nontata_promoters
          type: katarinagresova/Genomic_Benchmarks_human_nontata_promoters
        metrics:
          - type: matthews_correlation
            value: 0.747
            name: Test MCC (seed 42)
          - type: accuracy
            value: 0.872
            name: Test accuracy (seed 42)
          - type: f1
            value: 0.878
            name: Test F1 (seed 42)
---

# Nucleotide Transformer v2 50M, fine-tuned on GenomicBenchmarks

Full weight-level fine-tuning of [InstaDeepAI/nucleotide-transformer-v2-50m-multi-species](https://huggingface.co/InstaDeepAI/nucleotide-transformer-v2-50m-multi-species) for binary DNA sequence classification on two [GenomicBenchmarks](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks) tasks. All parameters are updated rather than using LoRA or a frozen backbone, with a leakage-free train/validation/test protocol and multi-seed evaluation.

> ### Weights are not redistributed: this is a card and code release
>
> The base model is licensed **CC-BY-NC-SA-4.0** (non-commercial, share-alike), and any fine-tuned derivative inherits those terms. To respect that license, the trained checkpoints are **not** hosted here. The full training and evaluation code is open source and the run is reproducible, with every seed, split, and hyperparameter fixed.
>
> **Code, reproduction steps, figures, and the full model card: [github.com/ankurgenomics/genome-ft](https://github.com/ankurgenomics/genome-ft)**
>
> Running the documented commands regenerates the checkpoints described below.

## Results

All numbers are on the held-out **test set, evaluated exactly once** on the best checkpoint (selected by validation MCC). The base model is the pretrained backbone with an untrained classification head, measured under the identical pipeline.

### Enhancers: `human_enhancers_cohn` (6,948 test examples, 3 seeds: 42, 0, 123)

| Model | Accuracy | F1 | MCC |
|-------|----------|-----|-----|
| Base (pretrained backbone, untrained head) | 0.499 | 0.021 | −0.009 |
| **Fine-tuned (this work)** | **0.735 ± 0.004** | **0.745 ± 0.016** | **0.478 ± 0.003** |

Per-seed test MCC: 0.481, 0.478, 0.474 (σ = 0.003), indicating a stable result across initialisations.

### Promoters: `human_nontata_promoters` (9,034 test examples, seed 42)

| Model | Accuracy | F1 | MCC |
|-------|----------|-----|-----|
| Base (pretrained backbone, untrained head) | 0.451 | 0.064 | −0.044 |
| **Fine-tuned (this work)** | **0.872** | **0.878** | **0.747** |

### Where it lands (published reference numbers, accuracy on enhancers)

| Model | Accuracy | Note |
|-------|----------|------|
| GB-CNN (Grešová et al. 2023) | 0.69 | published reference |
| DNABERT (Ji et al. 2021) | 0.706 | published reference |
| **This work — NT-v2 50M full fine-tune** | **0.735** | measured here, 3 seeds |
| HyenaDNA tiny-1k (Nguyen et al. 2023) | 0.74 | published reference |
| NT-v2 500M (Dalla-Torre et al. 2023) | 0.776 | published reference |
| DNABERT-2 (Zhou et al. 2023) | 0.785 | published reference |

On enhancers, the 50M model performs above the published CNN and DNABERT baselines and below the larger transformers, consistent with its parameter count. It does not outperform the 500M models, and is not intended to.

## How it was trained

- **Method:** full fine-tuning of all 53.8M parameters with a 2-layer classification head
- **Optimizer:** AdamW, learning rate 1e-5, weight decay 0.01
- **Schedule:** linear warmup (300 steps) with cosine decay, gradient clipping (max-norm 1.0)
- **Augmentation:** reverse-complement
- **Protocol:** 15% of training data held out for validation; checkpoint selected by validation MCC; test set evaluated once; 3 seeds for the primary task
- **Best epoch:** 1–2, after which validation MCC declines; the low learning rate and validation-based selection limit overfitting beyond the early epochs

## Intended use & limitations

- **Intended use:** research and educational demonstration of adapting a genomic foundation model at the weight level on standard DNA classification benchmarks.
- **Not for:** clinical, diagnostic, or any commercial use (the CC-BY-NC-SA-4.0 license forbids commercial use).
- **Scope:** two GenomicBenchmarks tasks only; results reflect those datasets and may not transfer to other genomic tasks, species, or sequence lengths.

## License & attribution

- **Fine-tuned weights:** CC-BY-NC-SA-4.0 (inherited from the base model; non-commercial, share-alike, attribution).
- **Training/evaluation code:** MIT (see the GitHub repository).
- **Attribution:** base model by InstaDeep; benchmark datasets by the GenomicBenchmarks authors (Grešová et al. 2023).

## Citation / related work

For the broader research context, see Fesser, Zhang, Li, Zitnik et al., *How Post-Training Shapes Biological Reasoning Models* ([arXiv:2606.16517](https://arxiv.org/abs/2606.16517), 2026). Its Finding 1, that supervised fine-tuning improves in-domain accuracy while out-of-domain performance peaks early and declines, has an in-domain counterpart in this work: validation MCC peaked at epoch 1–2 and then declined, which motivates the use of a low learning rate and validation-based checkpoint selection rather than longer training. This is a single controlled fine-tune, not a comparable research program; the connection is one of shared principle, not scope.

---

Developed by **Ankur Sharma**.

*This is a personal open-source project, developed independently in a personal capacity. It is not affiliated with, endorsed by, or representative of any current or former employer, and uses only public models and public benchmark datasets. All views and results are the author's own.*
