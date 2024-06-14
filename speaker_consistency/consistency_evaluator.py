try:
    import modelscope
except ImportError:
    print("The 'modelscope' module is not installed. Please install it using 'pip install modelscope'.")
    exit(1)

try:
    import sklearn
except ImportError:
    print("The 'sklearn' module is not installed. Please install it using 'pip install scikit-learn'.")
    exit(1)

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as transforms
from modelscope.pipelines import pipeline
from tqdm import tqdm


def resample_audio(waveform, original_sample_rate, target_sample_rate=16000):
    if original_sample_rate != target_sample_rate:
        resampler = transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate)
        waveform = resampler(waveform)
    return waveform


def load_and_resample_wav(file_path, target_sample_rate=16000):
    waveform, original_sample_rate = torchaudio.load(file_path)
    waveform = resample_audio(waveform, original_sample_rate, target_sample_rate)
    if waveform.shape[0] > 1:
        waveform = waveform[0:1, :]
    return waveform.squeeze().numpy()


def compute_cos_similarity(emb1, emb2):
    emb1 = torch.from_numpy(emb1).unsqueeze(0) if isinstance(emb1, np.ndarray) else emb1
    emb2 = torch.from_numpy(emb2).unsqueeze(0) if isinstance(emb2, np.ndarray) else emb2
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    return cos(emb1, emb2).item()


def calculate_rank(mean_pair_col, std_pair_col):
    std_pair_min = std_pair_col.min()
    std_pair_max = std_pair_col.max()
    norm_std_pair_col = (std_pair_col - std_pair_min) / (std_pair_max - std_pair_min)
    rank_col = 0.7 * mean_pair_col + 0.3 * (1 - norm_std_pair_col)
    return rank_col


sv_pipeline = pipeline(
    task='speaker-verification',
    model='iic/speech_eres2netv2_sv_zh-cn_16k-common',
    model_revision='v1.0.1'
)


def process_audio_files(root_dir):
    results = []
    for speaker_id in tqdm(os.listdir(root_dir)):
        speaker_path = os.path.join(root_dir, speaker_id)
        if os.path.isdir(speaker_path):
            for dataset_name in os.listdir(speaker_path):
                dataset_dir = os.path.join(speaker_path, dataset_name)
                if os.path.isdir(dataset_dir):
                    wav_files = [f for f in os.listdir(dataset_dir) if f.endswith('.wav')]
                    wav_paths = [os.path.join(dataset_dir, file_path) for file_path in wav_files]
                    waveforms = [load_and_resample_wav(file_path) for file_path in wav_paths]
                    result = sv_pipeline(waveforms, output_emb=True)
                    embeddings = result['embs']
                    # 计算平均嵌入 并计算均值和标准差
                    mean_embedding = np.mean(embeddings, axis=0)
                    cos_similarities = [compute_cos_similarity(mean_embedding, emb) for emb in embeddings]
                    mean_avg = np.mean(cos_similarities)
                    std_avg = np.std(cos_similarities)
                    # 成对计算相似度 并计算均值和标准差
                    pairwise_similarities = []
                    for i in range(len(embeddings)):
                        for j in range(i + 1, len(embeddings)):
                            pairwise_similarities.append(compute_cos_similarity(embeddings[i], embeddings[j]))
                    mean_pair = np.mean(pairwise_similarities)
                    std_pair = np.std(pairwise_similarities)
                    results.append({
                        'id': str(speaker_id),
                        'data_set': dataset_name,
                        'mean_avg': mean_avg,
                        'std_avg': std_avg,
                        'mean_pair': mean_pair,
                        'std_pair': std_pair
                    })
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate speaker consistency.')
    parser.add_argument('--dir', type=str, default='./test_audio',
                        help='Path to the input directory containing test audio files.')
    args = parser.parse_args()
    results = process_audio_files(args.dir)
    df = pd.DataFrame(results)
    df['id'] = df['id'].astype(str)
    # 将数据展开
    df_long = df.melt(id_vars=['id', 'data_set'], var_name='metric', value_name='value')
    df_long['metric'] = df_long['metric'] + '_' + df_long['data_set']
    df_pivot = df_long.pivot(index='id', columns='metric', values='value')
    for dataset_name in df['data_set'].unique():
        mean_pair_col = df_pivot[f'mean_pair_{dataset_name}']
        std_pair_col = df_pivot[f'std_pair_{dataset_name}']
        df_pivot[f'rank_{dataset_name}'] = calculate_rank(mean_pair_col, std_pair_col)
    df_pivot.to_csv('evaluation_results.csv', index_label='id')
    print(df_pivot)
