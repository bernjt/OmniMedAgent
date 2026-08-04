# OmniMedAgent

OmniMedAgent couples hierarchical clinical planning with reliability-weighted Dempster–Shafer evidence accumulation for multimodal diagnostic reasoning, risk stratification, report generation, treatment recommendation, and pathology analysis.

## Installation

Use Python 3.10 and CUDA 12.1.

```bash
conda env create -f environment.yml
conda activate omni-med-agent
pip install -e .
```

```bash
docker build -t omni-med-agent .
```

## Data

The verified access locations and agreements are listed in `datasets.txt`. MIMIC-IV, eICU-CRD, and HiRID require PhysioNet credentialing. CheXpert requires the Stanford AIMI agreement. TCGA is accessed through the GDC portal. Create a patient-separated CSV manifest with the fields in `code/omni_med_agent/data/manifest.py`. Raw clinical records and credentials must remain outside the repository.

The NIH ChestX-ray14 address stated in the manuscript returned HTTP 404 during release preparation and is omitted from `datasets.txt`. Add it only after the provider restores a canonical access endpoint.

## Training

The main configuration follows Supplementary Table S16: Qwen2.5-VL-7B, SigLIP-SO400M/14, 50 hypotheses, 100 focal elements, seven planning steps, PPO clip 0.2, GAE λ 0.95, conflict threshold 0.3, and 10,000 RL steps.

```bash
PYTHONPATH=code torchrun --nproc_per_node=8 -m omni_med_agent.cli.train --config configurations/main.yaml
```

Pretraining uses batch size 128 for three epochs at learning rate 2e-5. Supervised fine-tuning uses batch size 32 for five epochs at learning rate 1e-5. PPO uses batch size 16 at learning rate 5e-7. The reported full run used eight NVIDIA A100 80GB GPUs for approximately 480 GPU-hours.

## Evaluation

Primary evaluation uses mean AUROC for imaging diagnosis, accuracy for sequential diagnosis, clinical entity F1 for report generation, guideline concordance for treatment recommendation, and AUROC for risk and pathology tasks. Report mean and standard deviation over 15 seeds, 95% intervals from 10,000 bootstrap resamples, paired permutation and Wilcoxon tests, Cohen's d, continuous NRI, and IDI.

## Architecture

`code/omni_med_agent/evidence` contains sparse focal mass assignment, Dempster combination, conflict rejection, belief, plausibility, entropy, and clinical prior integration. `code/omni_med_agent/planning` contains the shared belief encoder, seven-strategy policy, 23-action sequence policy, adaptive termination, information-gain reward, generalized advantage estimation, and PPO objectives. Modality encoders, task heads, dataset specifications, training state, and statistical evaluation are isolated in their corresponding packages.

## Compute budget

Training requires eight A100 80GB GPUs under the reported configuration and approximately 480 aggregate GPU-hours. A single case takes about 4.2 seconds on one A100 80GB with a reported peak allocation of 18.8GB. Storage depends on accepted dataset versions and locally generated manifests; raw data are not copied into this project.

## Clinical use

This software is for retrospective research. It is not a medical device and must not be used to diagnose, treat, triage, or make autonomous decisions about patients.
