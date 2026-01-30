import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Feature builder per iteration 43 instructions

BASES = ['A', 'C', 'G', 'U']
BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}


def clean_seq(seq: str, length: int = 19) -> str:
    seq = (seq or '').upper().replace('T', 'U')
    cleaned = ''.join([ch if ch in BASES else 'N' for ch in seq])
    if len(cleaned) < length:
        cleaned = cleaned + 'N' * (length - len(cleaned))
    else:
        cleaned = cleaned[:length]
    return cleaned


def one_hot_seq(seq: str) -> np.ndarray:
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for i, ch in enumerate(seq):
        if ch in BASE_TO_IDX:
            arr[i, BASE_TO_IDX[ch]] = 1.0
    return arr.reshape(-1)


def interaction_features(si: str, mr: str):
    wc_set = {('A', 'U'), ('U', 'A'), ('G', 'C'), ('C', 'G')}
    wobble_set = {('G', 'U'), ('U', 'G')}
    wc = np.zeros(len(si), dtype=np.float32)
    wob = np.zeros(len(si), dtype=np.float32)
    mm = np.zeros(len(si), dtype=np.float32)
    for i, (a, b) in enumerate(zip(si, mr)):
        if a in BASES and b in BASES:
            pair = (a, b)
            if pair in wc_set:
                wc[i] = 1.0
            elif pair in wobble_set:
                wob[i] = 1.0
            else:
                mm[i] = 1.0
        else:
            mm[i] = 1.0
    total_wc = wc.sum()
    total_wob = wob.sum()
    total_mm = mm.sum()
    seed_slice = slice(1, 8)  # positions 2-8
    seed_wc = wc[seed_slice].sum()
    seed_wob = wob[seed_slice].sum()
    per_pos = np.concatenate([wc, wob, mm]).astype(np.float32)
    summary = np.array([total_wc, total_wob, total_mm, seed_wc, seed_wob], dtype=np.float32)
    return per_pos, summary


def kmer_counts(seq: str):
    mono = np.zeros(4, dtype=np.float32)
    for ch in seq:
        if ch in BASE_TO_IDX:
            mono[BASE_TO_IDX[ch]] += 1
    if len(seq) > 0:
        mono /= len(seq)
    di = np.zeros(16, dtype=np.float32)
    for i in range(len(seq) - 1):
        a, b = seq[i], seq[i + 1]
        if a in BASE_TO_IDX and b in BASE_TO_IDX:
            idx = BASE_TO_IDX[a] * 4 + BASE_TO_IDX[b]
            di[idx] += 1
    if len(seq) > 1:
        di /= (len(seq) - 1)
    return mono, di


def build_feature_matrix(df: pd.DataFrame, encoder: OneHotEncoder = None, fit_encoder: bool = False, artifacts_path: str = None):
    si_clean = df['siRNA'].apply(clean_seq)
    mr_clean = df['mRNA'].apply(clean_seq)

    seq_features = []
    inter_per_pos = []
    inter_summary = []
    kmer_feats = []
    for s, m in zip(si_clean, mr_clean):
        seq_features.append(np.concatenate([one_hot_seq(s), one_hot_seq(m)]))
        per_pos, summary = interaction_features(s, m)
        inter_per_pos.append(per_pos)
        inter_summary.append(summary)
        mono_si, di_si = kmer_counts(s)
        mono_mr, di_mr = kmer_counts(m)
        kmer_feats.append(np.concatenate([mono_si, di_si, mono_mr, di_mr]))

    seq_arr = np.vstack(seq_features)
    inter_arr = np.vstack(inter_per_pos)
    inter_sum_arr = np.vstack(inter_summary)
    kmer_arr = np.vstack(kmer_feats)

    drop_cols = ['siRNA', 'mRNA', 'extended_mRNA', 'efficiency', 'numeric_label', 'id', 'source', 'cell_line']
    numeric_cols = [c for c in df.columns if c not in drop_cols]
    numeric_arr = df[numeric_cols].astype(np.float32).to_numpy()

    cat_df = df[['source', 'cell_line']]
    if encoder is None:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    if fit_encoder:
        cat_arr = encoder.fit_transform(cat_df)
    else:
        cat_arr = encoder.transform(cat_df)

    feats = np.concatenate([seq_arr, inter_arr, inter_sum_arr, kmer_arr, numeric_arr, cat_arr], axis=1)

    feature_names = []
    feature_names += [f'siRNA_pos{i + 1}_{b}' for i in range(19) for b in BASES]
    feature_names += [f'mRNA_pos{i + 1}_{b}' for i in range(19) for b in BASES]
    feature_names += [f'inter_wc_pos{i + 1}' for i in range(19)]
    feature_names += [f'inter_wobble_pos{i + 1}' for i in range(19)]
    feature_names += [f'inter_mismatch_pos{i + 1}' for i in range(19)]
    feature_names += ['total_wc', 'total_wobble', 'total_mismatch', 'seed_wc', 'seed_wobble']
    feature_names += [f'si_mono_{b}' for b in BASES]
    feature_names += [f'si_di_{i}' for i in range(16)]
    feature_names += [f'mr_mono_{b}' for b in BASES]
    feature_names += [f'mr_di_{i}' for i in range(16)]
    feature_names += numeric_cols
    feature_names += encoder.get_feature_names_out(['source', 'cell_line']).tolist()

    if fit_encoder and artifacts_path:
        artifact = {
            'categories': [cats.tolist() for cats in encoder.categories_],
            'category_feature_names': encoder.get_feature_names_out(['source', 'cell_line']).tolist(),
            'numeric_cols': numeric_cols,
            'feature_names': feature_names,
        }
        Path(artifacts_path).write_text(json.dumps(artifact, indent=2))

    return feats.astype(np.float32), feature_names, encoder


if __name__ == '__main__':
    # Simple self-test on train.csv to ensure shapes
    train_path = '/workspace/datasets/siRBench/train.csv'
    df = pd.read_csv(train_path)
    feats, names, enc = build_feature_matrix(df, fit_encoder=True, artifacts_path='./feature_artifacts.json')
    print('features', feats.shape, 'num_names', len(names))
