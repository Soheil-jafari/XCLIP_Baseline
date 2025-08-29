X-CLIP Baseline Experiments for Paper
This folder contains a complete, robust, and efficient implementation for running X-CLIP baseline experiments as required for your paper. The code is structured to be easy to run and directly uses the paths from your server configuration.

1. Setup
First, create a dedicated conda environment and install the required packages.

# Navigate to this directory
cd xclip_paper_baselines

# Create and activate a new conda environment
conda create -n xclip python=3.9 -y
conda activate xclip

# Install PyTorch with CUDA support (adjust version if needed for your server)
pip install torch torchvision --extra-index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Install the rest of the dependencies
pip install -r requirements.txt

2. Running the Experiments
There are three main scripts, one for each experiment. They will create timestamped output folders inside /home/240331715/data/project_folder/Language-Guided-Endoscopy-Localization/outputs/xclip_baselines and save checkpoints in a similar checkpoints directory.

Experiment 1: Zero-Shot Evaluation
This runs the pre-trained X-CLIP model on your test set without any training.

python run_zeroshot.py

Experiment 2: Linear-Probe (Few-Shot)
This freezes the X-CLIP backbone and trains a small linear head. It will first train the head and then automatically run a full evaluation on the test set.

# You can adjust hyperparameters like epochs, batch size, and learning rate
python run_linear_probe.py --epochs 15 --lr 1e-3

Experiment 3: Full Fine-Tuning
This fine-tunes the entire model using a contrastive loss on your training data. This is the most computationally intensive experiment. It will train the model and then evaluate it on the test set.

# Adjust hyperparameters as needed for your experiments
python run_finetune.py --epochs 5 --batch-size 16 --lr 1e-5

Outputs
For each run, a metrics.json and a human-readable metrics.txt file will be generated in the corresponding output directory. These files contain the Macro AUROC and AP scores you need for your paper's comparison tables.
