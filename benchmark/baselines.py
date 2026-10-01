"""Classical design references and train-only linear baselines for frozen folds."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REFERENCE = Path(__file__).parent / 'reference_data'
ALPHAS = (.001, .01, .1, 1., 10., 100., 1000., 10000.)


def guides(frame):
    sequences = frame.siRNA.str.upper().str.replace('T', 'U').tolist()
    if any(len(seq) != 19 or set(seq) - set('ACGU') for seq in sequences):
        raise ValueError('Baselines require unambiguous 19-nt guide strands')
    return sequences


def iscore(sequences):
    coefficients = pd.read_csv(REFERENCE / 'iscore_2007_antisense.csv').set_index('position')
    return np.array([sum(coefficients.loc[i+1, base] for i, base in enumerate(seq))
                     for seq in sequences])


def uitei_functional(sequences):
    """The four functionality conditions documented by siDirect2.1, not its off-target pipeline."""
    return np.array([seq[0] in 'AU' and seq[-1] in 'GC'
                     and sum(base in 'AU' for base in seq[:7]) >= 4
                     and re.search('[GC]{10}', seq) is None for seq in sequences], dtype=float)


def sequence_features(frame):
    return np.array([[base == nucleotide for base in seq for nucleotide in 'ACGU']
                     for seq in guides(frame)], dtype=float)


def ridge_fit(train_features, validation_features, train_labels, validation_labels):
    scores = []
    models = []
    for alpha in ALPHAS:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver='svd'))
        model.fit(train_features, train_labels)
        scores.append(r2_score(validation_labels, model.predict(validation_features)))
        models.append(model)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite validation R² for ridge selection')
    selected = int(np.argmax(scores))
    return models[selected], {'selected_alpha':ALPHAS[selected],
                              'validation_r2_by_alpha':dict(zip(map(str, ALPHAS), scores))}


def predict_baselines(train, validation, evaluations, feature_columns):
    train_y = train.efficiency.to_numpy()
    val_y = validation.efficiency.to_numpy()
    result = {'training_mean':{name:np.full(len(frame), train_y.mean())
                              for name, frame in evaluations.items()}}
    selection = {'training_mean':{'training_mean':float(train_y.mean())}}
    for name, transform in [('guide_ridge', sequence_features),
                            ('thermodynamic_ridge', lambda f:f[feature_columns].to_numpy(float)),
                            ('guide_thermodynamic_ridge', lambda f:np.column_stack([sequence_features(f), f[feature_columns].to_numpy(float)]))]:
        model, selection[name] = ridge_fit(transform(train), transform(validation), train_y, val_y)
        result[name] = {cohort:model.predict(transform(frame)) for cohort, frame in evaluations.items()}
    result['iscore_2007_fixed'] = {name:iscore(guides(frame))/100 for name, frame in evaluations.items()}
    for name, score in [('iscore_2007_calibrated', iscore), ('uitei_sidirect_calibrated', uitei_functional)]:
        model = LinearRegression().fit(score(guides(train)).reshape(-1,1), train_y)
        selection[name] = {'intercept':float(model.intercept_), 'coefficient':float(model.coef_[0])}
        result[name] = {cohort:model.predict(score(guides(frame)).reshape(-1,1)) for cohort, frame in evaluations.items()}
    return result, selection


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--feature-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a new, empty baseline output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    columns = json.loads(args.feature_manifest.read_text())['features']
    matrix = pd.read_csv(args.protocol/'run_matrix.csv').drop_duplicates(['axis','fold'])
    predictions, selections = [], []
    hela = pd.read_csv(args.protocol/'hela_full.csv')
    for row in matrix.itertuples(index=False):
        train, val, test = (pd.read_csv(path) for path in [row.train,row.val,row.test])
        evaluations = {'test':test, 'hela_full':hela}
        fitted, parameters = predict_baselines(train,val,evaluations,columns)
        for model, cohorts in fitted.items():
            for cohort, values in cohorts.items():
                frame = evaluations[cohort]
                if not np.isfinite(values).all():
                    raise ValueError(f'Nonfinite {model} predictions')
                predictions.append(pd.DataFrame({'tool':model,'axis':row.axis,'fold':row.fold,
                    'training_seed':-1,'cohort':cohort,'record_id':frame.record_id,
                    'label':frame.efficiency,'pred_label':values}))
        selections.append({'axis':row.axis,'fold':int(row.fold),'models':parameters})
        print(row.axis,row.fold,'complete',flush=True)
    pd.concat(predictions,ignore_index=True).to_csv(args.output/'predictions.csv',index=False)
    (args.output/'selection.json').write_text(json.dumps(selections,indent=2)+'\n')
    manifest = {'training_policy':'Deterministic baselines fitted on training rows only; ridge alpha chosen on validation R²; no train+validation refit or prediction clipping.',
                'training_seed':-1,'seed_note':'One deterministic fit per partition; never count repeated identical predictions as seed replication.',
                'iscore_note':'Historical frozen coefficients trained on2431Huesken rows;2361benchmark guides overlap. Fixed score/100 is a design index, not calibrated efficacy; calibrated variant fits a train-only affine mapping.',
                'uitei_note':'Only siDirect2.1 functionality conditions, followed by train-only affine calibration; no seed-Tm or transcriptome off-target filters.',
                'feature_columns':columns,'ridge_alpha_grid':ALPHAS,
                'inputs':{str(path):sha256(path) for path in [args.protocol/'manifest.json', args.protocol/'run_matrix.csv',args.feature_manifest]},
                'code_sha256':sha256(Path(__file__)),
                'reference_sha256':{str(path.name):sha256(path) for path in REFERENCE.iterdir() if path.is_file()},
                'predictions_sha256':sha256(args.output/'predictions.csv')}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    main()
