"""Frozen iteration-16 architecture; no validation-fitted ensemble parameters."""
from sequence_model import Model as SequenceCNN
from thermo_model import Model as ThermodynamicCNN

CNN_TRAINING = dict(learning_rate=0.001, betas=(0.9,0.999), eps=1e-8,
    weight_decay=0.01, batch_size=64, drop_last=False, gradient_clip=1.0,
    max_epochs=100, patience=20, scheduler_T_max=100, scheduler_eta_min=1e-5)
CATBOOST_PARAMETERS = dict(iterations=2500, learning_rate=0.03, depth=3,
    l2_leaf_reg=20, loss_function='RMSE', eval_metric='R2', task_type='CPU',
    boosting_type='Plain', grow_policy='SymmetricTree', bootstrap_type='Bayesian',
    bagging_temperature=1, random_strength=1, rsm=1, border_count=64,
    boost_from_average=True, od_type='Iter', od_wait=150, use_best_model=True,
    thread_count=16, allow_writing_files=False, verbose=False)

def make_sequence_model(training_label_mean):
    return SequenceCNN(p=0.4, mean=training_label_mean)

def make_thermodynamic_model(training_label_mean):
    return ThermodynamicCNN(h=16, p_thermo=0.5, mean=training_label_mean)

def make_tree_model(seed=0):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(**CATBOOST_PARAMETERS, random_seed=int(seed))

def combine_predictions(sequence, thermodynamic, tree):
    import numpy as np
    arrays=[np.asarray(x,dtype=np.float64) for x in (sequence,thermodynamic,tree)]
    if any(x.ndim != 1 or x.shape != arrays[0].shape or not np.isfinite(x).all() for x in arrays):
        raise ValueError('Expected three matching finite prediction vectors')
    return (arrays[0]+arrays[1]+arrays[2])/3.0
