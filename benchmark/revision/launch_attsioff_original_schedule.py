#!/usr/bin/env python3
"""Launch the isolated released-schedule entry point in the frozen AttSiOff image."""
import argparse
from pathlib import Path
import sys

scripts = Path(__file__).resolve().parents[1] / 'competitors/scripts'
sys.path.insert(0, str(scripts))
from runner import repo_root, run_docker
from train import rewrite_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tool', choices=['attsioff'], required=True)
args, forwarded = parser.parse_known_args()
host_root = repo_root(str(scripts))
run_docker('attsioff', '../../../revision/attsioff_original_schedule.py', rewrite_args(forwarded, host_root), host_root)
