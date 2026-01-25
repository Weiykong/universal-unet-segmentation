import os
import subprocess
import sys
import argparse

def setup_directories():
    """Ensure the project structure is ready."""
    dirs = ['data/images', 'data/masks', 'models', 'output']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✅ Directory structure verified.")

def run_pipeline():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Universal U-Net Pipeline Control")
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['all', 'train', 'inference'],
                        help="Choose 'train' to skip inference, 'inference' to skip training, or 'all'.")
    
    args = parser.parse_args()
    setup_directories()
    
    # --- Phase 1: Training ---
    if args.mode in ['all', 'train']:
        print("\n--- Phase 1: Training ---")
        try:
            subprocess.run([sys.executable, "src/train.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed: {e}")
            return
    else:
        print("\n⏭️ Skipping Training (Inference Only Mode)")

    # --- Phase 2: Inference ---
    if args.mode in ['all', 'inference']:
        print("\n--- Phase 2: Inference / Probability Map Generation ---")
        try:
            subprocess.run([sys.executable, "src/infrence.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Inference failed: {e}")
            return
    else:
        print("\n⏭️ Skipping Inference (Training Only Mode)")

    print("\n🎉 Process complete!")

if __name__ == "__main__":
    run_pipeline()