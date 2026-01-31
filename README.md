# X-CLIP Baseline Experiments: Temporal Localization in Surgery

[cite_start]This repository provides a robust implementation of **X-CLIP** as a benchmark for language-guided temporal action localization (TAL) on endoscopic video[cite: 72, 748]. [cite_start]By utilizing multi-grained temporal contrastive learning, this baseline evaluates the transferability of general-purpose video-language models to the specialized surgical domain[cite: 747, 782].

## 🏗️ Architecture & Differentiation

[cite_start]Unlike standard image-based models (like vanilla CLIP), **X-CLIP** is designed specifically for video-text alignment[cite: 782]. [cite_start]It was chosen as a critical baseline because it moves beyond processing independent frames by introducing temporal awareness[cite: 747].

**Key Model Components:**
* [cite_start]**Multi-Grained Temporal Contrastive Learning**: Captures both frame-level details and high-level temporal dynamics across video segments[cite: 782].
* [cite_start]**Video-Specific Encoder**: Uses a message-passing mechanism between frame tokens to exchange temporal information before global pooling[cite: 782].
* [cite_start]**Cross-Modal Fusion**: Aligns visual representations of 16-frame clips with clinical descriptions in a shared embedding space[cite: 784].



## 💻 Hardware & Resource Management

[cite_start]To handle the high-resolution, long-form nature of the **Cholec80 dataset**, training was optimized for high-performance research infrastructure[cite: 343, 593].

* [cite_start]**GPU Infrastructure**: Executed on **NVIDIA RTX A6000 GPUs** with **48GB of VRAM**[cite: 592].
* [cite_start]**VRAM Optimization**: Utilized **Mixed-Precision (FP16)** via PyTorch’s `autocast` to reduce the memory footprint and accelerate training throughput[cite: 598].
* [cite_start]**Compute Strategy**: Implemented **Gradient Accumulation** to simulate stable batch sizes (effective size of 192) within hardware limits[cite: 665, 692].
* [cite_start]**Session Persistence**: Managed long-running jobs (up to 17 hours per epoch) using **tmux** to ensure stability against network disconnections[cite: 598, 805].

## 📊 Baseline Comparison Results

[cite_start]While X-CLIP is a state-of-the-art general video model, results highlight a significant **domain gap** when applied to surgical endoscopy[cite: 872, 907].

| Metric | X-CLIP Baseline | **Proposed Framework (MSc Thesis)** |
| :--- | :--- | :--- |
| **AUROC** (Test) | [cite_start]0.50 [cite: 795, 967] | [cite_start]**0.93** [cite: 837, 911] |
| **AUPRC** (Test) | [cite_start]0.07 [cite: 795, 967] | [cite_start]**0.89** [cite: 837, 911] |
| **mAP@0.5** | [cite_start]0.00 [cite: 797, 967] | [cite_start]**0.29\*** [cite: 864, 990] |

[cite_start]**Technical Analysis**: X-CLIP showed "okayish" trends on validation ($AUROC \approx 0.72$) but collapsed on the test set ($AUROC \approx 0.50$) due to visual domain shift[cite: 964, 968]. [cite_start]This proves that standard video-language models struggle to generalize to the subtle, continuous procedural nature of surgery without domain-specific tailoring[cite: 872, 923].

## 🚀 Setup & Usage

### 1. Environment Configuration
[cite_start]Create a dedicated conda environment to manage dependencies[cite: 596]:

```bash
# Navigate to this directory
cd xclip_paper_baselines

# Create and activate a new conda environment
conda create -n xclip python=3.9 -y
conda activate xclip

# Install PyTorch with CUDA support (for university ML server)
pip install torch torchvision --extra-index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install -r requirements.txt
2. Running Experiments
The scripts generate timestamped outputs and save checkpoints in the configured server directories.


Experiment 1: Zero-Shot Evaluation Runs the pre-trained X-CLIP model on the test set without training.

Bash
python run_zeroshot.py

Experiment 2: Linear-Probe (Few-Shot) Freezes the X-CLIP backbone and trains a task-specific linear head.

Bash
python run_linear_probe.py --epochs 15 --lr 1e-3

Experiment 3: Full Fine-Tuning Fine-tunes the entire model using contrastive loss on Cholec80 training data.

Bash
python run_finetune.py --epochs 5 --batch-size 16 --lr 1e-5
📂 Repository Structure
train_xclip.py: Main training entry point.


eval_xclip.py: Validation and test metric script.

infer_xclip.py: Real-world inference for query testing.

xclip_package/: Core X-CLIP library files.


project_config.py: Global settings and hyperparameter management.
