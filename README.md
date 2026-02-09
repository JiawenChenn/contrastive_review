# Contrastive Dimension Reduction: A Systematic Review

This repository contains code and data for the paper:

**Contrastive Dimension Reduction: A Systematic Review**
[https://arxiv.org/abs/2510.11847](https://arxiv.org/abs/2510.11847)

## Repository Structure

```
.
├── mnist/                  # MNIST dataset experiments
├── protein/                # Mouse protein expression dataset experiments
├── small_molecule/         # Small molecule (Mixseq) dataset experiments
└── utils/                  # Shared utility code
    ├── CCUR/               # Contrastive CUR decomposition
    ├── CFS/                # Contrastive Feature Selection
    ├── clvm/               # Contrastive Latent Variable Model
    └── contrastive_vae/    # Contrastive VAE
```

Each dataset folder contains:
- `data/` — Input data files
- Individual scripts for each contrastive method

## Methods

| Method | Scripts | Language |
|--------|---------|----------|
| PCA | `PCA_*.py` | Python |
| CPCA (Contrastive PCA) | `CPCA_*.py` | Python |
| PCPCA (Probabilistic Contrastive PCA) | `PCPCA_*.py` | Python |
| GCPCA (Generalized Contrastive PCA) | `GCPCA_*.py` | Python |
| scPCA (Sparse Contrastive PCA) | `scPCA_*.R` | R |
| cLVM (Contrastive Latent Variable Model) | `CLVM_*.py` | Python |
| CPLVM (Contrastive Poisson LVM) | `CPLVM_*.py` | Python |
| CVAE (Contrastive VAE) | `CVAE_*.py` | Python |
| ContrastiveVI | `CVI_*.py` | Python |
| CCUR (Contrastive CUR) | `CCUR_*.py` | Python |
| CFS (Contrastive Feature Selection) | `CFS_*.py` | Python |
| BasCoD (Basis of Contrastive Directions) | `BasCoD_*.py` | Python |
| CIR (Contrastive Inverse Regression) | `CIR_*.py` | Python |

## Datasets

- **MNIST** — Handwritten digit images (foreground: digits 2 & 3, background: digits 0 & 1)
- **Mouse Protein Expression** — Protein expression levels in mouse cortex (Higuera et al., 2015)
- **Small Molecule (Mixseq)** — Single-cell gene expression under drug perturbation (McFarland et al., 2020)

## Citation

```bibtex
@article{hawke2025contrastive,
  title={Contrastive dimension reduction: A systematic review},
  author={Hawke, Sam and Zhang, Eric and Chen, Jiawen and Li, Didong},
  journal={arXiv preprint arXiv:2510.11847},
  year={2025}
}
```
