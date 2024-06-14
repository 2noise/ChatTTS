[English README](README.en.md) | [中文简体](README.md)

# 评估说话人音色稳定性

找到音色稳定的说话人。主要代码包括音频生成、稳定性评估两个脚本。

评估基于通义实验室 [ERes2NetV2 说话人识别模型](https://modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/summary)

## 文件结构

```
speaker_consistency/
├── audio_generator.py
├── consistency_evaluator.py
├── README.md
├── requirements.txt
└── test_data.yaml
```


- `audio_generator.py`: 生成测试音频的脚本
- `consistency_evaluator.py`: 评估稳定性的脚本
- `test_data.yaml`: 生成音频的文本数据

## 安装

请确保您已经安装了 ChatTTS 项目 并正常运行。

```bash
pip install -r ./speaker_consistency/requirements.txt
```

## 使用说明

### 1. 生成测试音频

运行`audio_generator.py`脚本生成测试音频：

```bash
cd speaker_consistency
python audio_generator.py
```
### 2. 评估说话人稳定性

运行`consistency_evaluator.py`脚本进行评估：

```bash
cd speaker_consistency
python consistency_evaluator.py
```

评估结果将会保存在`evaluation_results.csv`文件中。

### 3. 配置文件

您可以通过编辑`test_data.yaml`文件来修改测试文本。


## 命令行参数说明

### 1. 生成脚本（`audio_generator.py`）

```bash
python audio_generator.py --dir <output_directory> --num <number_of_speakers> --ds <dataset_yaml_file>
```

参数说明：

- `--dir`：存储生成测试音频文件的目录。默认为 `./test_audio`。
- `--num`：生成的随机说话人数目。默认为 `10`。
- `--ds`：数据集YAML文件的路径。默认为 `test_data.yaml`。
- 
### 2. 评估脚本（`consistency_evaluator.py`）

```bash
python consistency_evaluator.py --dir <output_directory>
```

参数说明：

- `--dir`：输入目录的路径，包含需要进行一致性评估的测试音频文件。默认为 `./test_audio`。

## Rank 指标计算说明

在评估说话人稳定性时，通过计算每对音频片段嵌入向量之间的余弦相似度，得到相似度的均值和标准差。
对标准差进行归一化处理后，`rank` 指标中均值占70%的权重，标准差占30%的权重。
`rank` 指标越高，音频片段的一致性越好。


## 输出样本

| id   | ... | rank_TestA | rank_TestB |
|------|-----|------------|------------|
| 0000 | ... | 0.802779   | 0.809263   |
| 0001 | ... | 0.858448   | 0.773149   |
| 0002 | ... | 0.763376   | 0.779981   |

在所有样本集上得分都高的说话人相对更稳定，可以根据自己的使用场景调整测试集。