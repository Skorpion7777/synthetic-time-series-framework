# README

## Overview

This repository implements a framework for comparing synthetically generated time series data against traditionally anonymized data, focusing on electrocardiogram (ECG) datasets. The project evaluates the trade-offs between privacy and utility using advanced anonymization and synthetic data generation techniques.

## Repository Structure

- **Evaluation & Plots**: Scripts and plots for dataset evaluation.
- **TTS-CGAN**: Files related to the Time-Series Conditional Generative Adversarial Network for synthetic data generation by [imics-lab](https://github.com/imics-lab/tts-cgan)
- **kp-anonymization**: Implementations of the (k, P)-anonymity method for dataset anonymization by [jeorjebot](https://github.com/jeorjebot/kp-anonymity)
- **dataset_download.ipynb**: Jupyter notebook for downloading the ECG dataset.
- **preprocess_dataset.ipynb**: Jupyter notebook for data preprocessing.

## Getting Started

### Prerequisites

Ensure Python and Jupyter Notebook are installed. Install necessary packages:

```bash
pip install -r requirements.txt
```
### Sensitive Dataset

The Datasets used are the following from the [ECG PhysioNet 2021 Challenge](https://paperswithcode.com/dataset/physionet-challenge-2021)

1. China Physiological Signal Challenge (CPSC) 2018
2. St. Petersburg INCART 12-lead Arrhythmia Database
3. Physikalisch Technische Bundesanstalt (PTB) Diagnostic ECG Database
4. PTB-XL Database
5. Georgia database
6. An undisclosed American database

Alternatively, the processed dataset can be loaded via my [Drive](https://drive.google.com/drive/folders/1LQZEKvy_Xt_VhwqyQXXpwzrfPWQYEG91?usp=drive_link)

### Anonymized Dataset

### Synthetic Dataset
We can train the dataset using:
```bash
python mitbih_Train_CGAN.py
```
This Python file is a pre-configured training command with all arguments in place. Check out the file for details. Depending on computational ressources it is especially usefull to adjust the gen_bs, dis_bs and batch_size arguments. It generates a checkpoint file in the logs, which is our trained model and we can use down the line.
Later we can simply generate synthetic data using such a command:

```bash
syn_mitbih(n_samples=800, reshape=True)
```
Some transformation is done on it to be able to export it as a csv and use it for later evaluation down the line.

### Visual Evaluation

### Utility Evaluation

### Privacy Assessment
