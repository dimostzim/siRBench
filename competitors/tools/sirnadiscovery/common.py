import json
import os

import numpy as np
import pandas as pd
import stellargraph as sg


def load_utils(src_root):
    import sys
    sys.path.insert(0, src_root)
    import utils
    return utils


def load_params(path):
    with open(path, 'r') as f:
        return json.load(f)


def _standardize_sequences(df):
    df = df.copy()
    df['siRNA_seq'] = df['siRNA_seq'].astype(str).str.replace('U', 'T', regex=True)
    df['mRNA_seq'] = df['mRNA_seq'].astype(str).str.replace('U', 'T', regex=True)
    return df


def build_graph(df_all, preprocess_dir, ago2_dir, params, src_root):
    utils = load_utils(src_root)
    data = _standardize_sequences(df_all)

    sirna_onehot = [utils.obtain_one_hot_feature_for_one_sequence_1(seq, params["sirna_length"]) for seq in data['siRNA_seq']]
    sirna_onehot = pd.DataFrame(sirna_onehot, index=list(data['siRNA']))

    mrna_onehot_temp = data.loc[:, ['mRNA', 'mRNA_seq']].drop_duplicates(subset="mRNA")
    mrna_onehot = [utils.obtain_one_hot_feature_for_one_sequence_1(seq, params["max_mrna_len"]) for seq in mrna_onehot_temp['mRNA_seq']]
    mrna_onehot = pd.DataFrame(mrna_onehot, index=list(mrna_onehot_temp['mRNA']))

    trans_table = str.maketrans('ATCG', 'TAGC')
    data['match_pos'] = [seq[::-1].upper().translate(trans_table) for seq in data['siRNA_seq']]
    data['match_pos'] = data.apply(lambda row: row['mRNA_seq'].index(row['match_pos']), axis=1)

    sirna_pos_encoding = [utils.get_pos_embedding_sequence(num, params["sirna_length"], params["dmodel"]) for num in data['match_pos']]
    sirna_pos_encoding = pd.DataFrame(sirna_pos_encoding, index=list(data['siRNA']))

    sirna_thermo_feat = [utils.cal_thermo_feature(seq.replace('T', 'U')) for seq in data['siRNA_seq']]
    sirna_thermo_feat = pd.DataFrame(sirna_thermo_feat).reset_index(drop=True)
    sirna_thermo_feat = pd.concat([
        data['siRNA'].reset_index(drop=True),
        data['mRNA'].reset_index(drop=True),
        sirna_thermo_feat,
    ], axis=1)
    sirna_thermo_feat['index'] = sirna_thermo_feat['siRNA'] + '_' + sirna_thermo_feat['mRNA']
    sirna_thermo_feat = sirna_thermo_feat.set_index('index').drop(columns=['siRNA', 'mRNA'])

    con_feat = pd.read_csv(os.path.join(preprocess_dir, "con_matrix.txt"), header=None, index_col=0)
    con_feat = con_feat.reindex(sirna_thermo_feat.index).fillna(0.0)

    sirna_sfold_feat = pd.read_csv(os.path.join(preprocess_dir, "self_siRNA_matrix.txt"), header=None, index_col=0)
    sirna_sfold_feat = sirna_sfold_feat.reindex(sirna_onehot.index).fillna(0.0)

    mrna_sfold_feat = pd.read_csv(os.path.join(preprocess_dir, "self_mRNA_matrix.txt"), header=None, index_col=0)
    mrna_sfold_feat = mrna_sfold_feat.reindex(mrna_onehot.index).fillna(0.0)

    sirna_ago = pd.read_csv(os.path.join(ago2_dir, "siRNA_AGO2.csv"), index_col=0)
    sirna_ago = sirna_ago.reindex(sirna_onehot.index).fillna(0.0)

    mrna_ago = pd.read_csv(os.path.join(ago2_dir, "mRNA_AGO2.csv"), index_col=0)
    mrna_ago = mrna_ago.reindex(mrna_onehot.index).fillna(0.0)

    sirna_gc = [utils.countGC(seq) for seq in data['siRNA_seq']]
    sirna_gc = pd.DataFrame(sirna_gc, index=list(data['siRNA']))

    mrna_gc = [utils.countGC(seq) for seq in mrna_onehot_temp['mRNA_seq']]
    mrna_gc = pd.DataFrame(mrna_gc, index=list(mrna_onehot_temp['mRNA']))

    sirna_1_mer = pd.DataFrame([utils.single_freq(seq) for seq in data['siRNA_seq']])
    sirna_2_mers = pd.DataFrame([utils.double_freq(seq) for seq in data['siRNA_seq']])
    sirna_3_mers = pd.DataFrame([utils.triple_freq(seq) for seq in data['siRNA_seq']])
    sirna_4_mers = pd.DataFrame([utils.quadruple_freq(seq) for seq in data['siRNA_seq']])
    sirna_5_mers = pd.DataFrame([utils.quintuple_freq(seq) for seq in data['siRNA_seq']])

    sirna_k_mers = pd.concat([sirna_1_mer, sirna_2_mers, sirna_3_mers, sirna_4_mers, sirna_5_mers], axis=1)
    sirna_k_mers.index = data['siRNA']

    sirna_pos_scores = [utils.rules_scores(seq) for seq in data['siRNA_seq']]
    sirna_pos_scores = pd.DataFrame(sirna_pos_scores, index=list(data['siRNA']))

    sirna_pd = pd.concat([sirna_onehot, sirna_sfold_feat, sirna_ago, sirna_gc, sirna_k_mers, sirna_pos_scores], axis=1)
    mrna_pd = pd.concat([mrna_onehot, mrna_sfold_feat, mrna_ago, mrna_gc], axis=1)

    source = data['siRNA'] + "_" + data['mRNA']
    target_sirna = data['siRNA']
    target_mrna = data['mRNA']

    edges = pd.concat([
        pd.DataFrame({'source': source, 'target': target_sirna}),
        pd.DataFrame({'source': source, 'target': target_mrna}),
    ], ignore_index=True)

    sirna_pos_encoding.index = sirna_thermo_feat.index
    interaction_pd = pd.concat([sirna_thermo_feat, con_feat, sirna_pos_encoding], axis=1)

    graph = sg.StellarGraph({"siRNA": sirna_pd, "mRNA": mrna_pd, "interaction": interaction_pd},
                            edges=edges, source_column="source", target_column="target")
    return graph
