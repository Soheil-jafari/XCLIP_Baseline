# X-CLIP Baseline on Surgical Video (Cholec80)
Benchmarking a general-purpose video–text model (**X-CLIP**) for **language-guided temporal localization / moment retrieval** in endoscopic surgery.

> This repository is part of my MSc thesis work and is intended as a **strong, reproducible baseline** to quantify how well off-the-shelf video–text models transfer to the surgical domain.

---

## What this repo contains
- **Zero-shot inference**: run pretrained X-CLIP and score video clips against text queries.
- **Linear probe**: freeze backbone, train a lightweight head for the target task.
- **Full fine-tuning**: fine-tune X-CLIP on surgical clips + text prompts.
- Clean experiment entrypoints:
  - `run_zeroshot.py`
  - `run_linear_probe.py`
  - `run_finetune.py`
- Core model/components under: `xclip_module/`
- Central config: `xclip_config.py`

---

## Why X-CLIP (vs vanilla CLIP)?
- **CLIP is image–text**: it embeds independent frames without explicit temporal modeling.
- **X-CLIP is video–text**: it introduces temporal reasoning and multi-grained contrastive alignment designed for video–text matching.
- In this project, X-CLIP is used as a **baseline** to test whether general video–text alignment transfers to **surgical endoscopy**.

---

## Architecture (high-level)
- **Input**
  - Video is sampled into short clips (e.g., fixed number of frames per clip).
  - Text prompts represent the target surgical event/query.
- **Encoders**
  - A transformer-based **text encoder** produces a text embedding.
  - A CLIP/ViT-style **visual encoder** produces frame-level features.
- **Temporal module (X-CLIP)**
  - Adds **temporal interaction** across frame features before pooling.
  - Produces a clip-level **video embedding** that encodes motion/temporal context.
- **Similarity / Retrieval scoring**
  - Computes similarity between **video embedding** and **text embedding**.
  - Generates temporal scores used for retrieval / localization decisions.

---

## Hardware & resource-aware training
- Trained/evaluated using a GPU setup suitable for long surgical videos.
- Techniques used to fit training into VRAM and keep runs stable:
  - **Mixed precision (FP16)**
  - **Gradient accumulation**
  - Long-run management via **tmux**
- If you reproduce results, please report:
  - GPU model, VRAM, CUDA/PyTorch versions, and batch/clip settings.

---

## Results (baseline takeaway)
- X-CLIP is strong on general video–text benchmarks, but surgical endoscopy introduces a **domain gap**.
- In my thesis evaluation (Cholec80 temporal localization setting), the baseline generalization on the held-out test split was limited:
  - **AUROC ≈ 0.50**
  - **AUPRC ≈ 0.07**
  - **mAP@0.5 ≈ 0.00**
- Interpretation: validation can look reasonable, but performance may collapse on test due to **distribution shift** in surgical footage.

---

## Setup
- Create an environment and install dependencies:
  - `pip install -r requirements.txt`
- Install a CUDA-matching PyTorch build for your system.

---

## Data
- This repository does **not** ship Cholec80 data.
- Obtain the dataset separately and configure local paths in:
  - `xclip_config.py`
- Expected inputs typically include:
  - extracted frames or decoded clips
  - annotations/metadata depending on your localization setup

---

## Running experiments
- Zero-shot:
  - `python run_zeroshot.py`
- Linear probe:
  - `python run_linear_probe.py`
- Full fine-tuning:
  - `python run_finetune.py`
- Outputs (checkpoints/logs) follow paths defined in `xclip_config.py`.

---

## References
- X-CLIP: https://arxiv.org/abs/2207.07285
- CLIP: https://arxiv.org/abs/2103.00020
- Cholec80 (CAMMA): https://camma.unistra.fr/datasets/

---

## Citation
- If you use this repository, please cite X-CLIP/CLIP and (optionally) my MSc thesis.

```bibtex
@article{ma2022xclip,
  title={X-CLIP: End-to-End Multi-grained Contrastive Learning for Video-Text Retrieval},
  author={Ma, Yiwei and Xu, Guohai and Sun, Xiaoshuai and Yan, Ming and Zhang, Ji and Ji, Rongrong},
  journal={arXiv preprint arXiv:2207.07285},
  year={2022}
}

@article{radford2021clip,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and others},
  journal={arXiv preprint arXiv:2103.00020},
  year={2021}
}
