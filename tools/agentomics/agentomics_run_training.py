#!/usr/bin/env python3
"""
Agentomics Training Script
Trains a LightGBM model on siRBench data using the existing train_model.py pipeline.
"""
import json
import subprocess
import sys
from pathlib import Path

def main():
    """Main entry point - runs the existing train_model.py script."""
    base_dir = Path(__file__).resolve().parent
    train_script = base_dir / "train_model.py"

    if not train_script.exists():
        print(f"Error: Training script not found at {train_script}")
        sys.exit(1)

    print("Running Agentomics training using existing pipeline...")
    print(f"Training script: {train_script}")
    print("-" * 50)
    print("Training on train set...")

    try:
        # Run the existing train_model.py script
        result = subprocess.run([
            sys.executable, str(train_script)
        ], capture_output=False, text=True)

        if result.returncode == 0:
            print("-" * 50)
            print("Training completed successfully!")
        else:
            print(f"Training failed with exit code: {result.returncode}")
            sys.exit(result.returncode)

    except Exception as e:
        print(f"Failed to run training script: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()