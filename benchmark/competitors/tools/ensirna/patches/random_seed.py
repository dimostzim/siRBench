import os
import random

import numpy as np
import torch


def _get_seed():
    env = os.environ.get("ENSIRNA_SEED")
    if env is not None:
        try:
            return int(env)
        except Exception:
            pass
    return 12


SEED = _get_seed()


def setup_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
