[English README](README.en.md) | [中文简体](README.md)

# Speaker Consistency Evaluation

This project aims to identify speakers with stable voice timbre. The main components of the project are scripts for generating audio and evaluating consistency.

The evaluation is based on the ERes2NetV2 Speaker Recognition Model from Tongyi Laboratory [ERes2NetV2 Speaker Recognition Model](https://modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/summary)

## Directory Structure

```
speaker_consistency/
├── audio_generator.py
├── consistency_evaluator.py
├── README.md
├── requirements.txt
└── test_data.yaml
```

- `audio_generator.py`: Script for generating test audio
- `consistency_evaluator.py`: Script for evaluating stability
- `test_data.yaml`: Text data for generating audio

## Installation

Ensure that you have installed and are running the ChatTTS project.

```bash
pip install -r ./speaker_consistency/requirements.txt
```

## Usage Instructions

### 1. Generate Test Audio

Run the `audio_generator.py` script to generate test audio:

```bash
cd speaker_consistency
python audio_generator.py
```

### 2. Evaluate Speaker Stability

Run the `consistency_evaluator.py` script for evaluation:

```bash
cd speaker_consistency
python consistency_evaluator.py
```

The evaluation results will be saved in the `evaluation_results.csv` file.

### 3. Configuration File

You can modify the test text by editing the `test_data.yaml` file.

## Command Line Arguments

### 1. Audio Generator Script (`audio_generator.py`)

```bash
python audio_generator.py --dir <output_directory> --num <number_of_speakers> --ds <dataset_yaml_file>
```

Argument Descriptions:

- `--dir`: Directory to store the generated test audio files. Default is `./test_audio`.
- `--num`: Number of random speakers to generate. Default is `10`.
- `--ds`: Path to the dataset YAML file. Default is `test_data.yaml`.

### 2. Consistency Evaluator Script (`consistency_evaluator.py`)

```bash
python consistency_evaluator.py --dir <output_directory>
```

Argument Descriptions:

- `--dir`: Path to the input directory containing the test audio files for consistency evaluation. Default is `./test_audio`.

## Rank Metric Calculation

When evaluating speaker stability, the cosine similarity between the embedding vectors of each pair of audio segments is calculated to obtain the mean and standard deviation of the similarities.
After normalizing the standard deviation, the `rank` metric assigns a 70% weight to the mean and a 30% weight to the standard deviation.
The higher the `rank` metric, the better the consistency of the audio segments.

## Sample Output

| id   | ... | rank_TestA | rank_TestB |
|------|-----|------------|------------|
| 0000 | ... | 0.802779   | 0.809263   |
| 0001 | ... | 0.858448   | 0.773149   |
| 0002 | ... | 0.763376   | 0.779981   |