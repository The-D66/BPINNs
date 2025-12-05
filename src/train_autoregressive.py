import os
import sys
import json
import tensorflow as tf
from datetime import datetime

# Disable Mixed Precision to avoid API issues in custom loop
# tf.keras.mixed_precision.set_global_policy('mixed_float16')
# print("Mixed Precision Policy: mixed_float16")

# Disable GPU if causing issues (Optional, but recommended for stability on some Macs)
# Fix for macOS TensorFlow/OpenMP crashes
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# Disable XLA to prevent CUDA Kernel errors
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add src to path
sys.path.append(os.path.dirname(__file__))

from setup.param import Param
from setup.window_loader import WindowDataset
from networks.ConvFormer import ConvFormer
from algorithms.Seq2SeqTrainer import Seq2SeqTrainer
from utility import set_directory

def main():
    set_directory()
    
    # Configuration
    config_name = "autoregressive_sv"
    config_path = f"../config/{config_name}.json"
    
    print(f"Loading configuration from {config_path}")
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        config_dict = json.load(f)
        
    # Create Timestampped Output Directory
    timestamp = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
    output_dir = f"../outs/SaintVenant1D/Autoregressive_{timestamp}"
    print(f"Output Directory: {output_dir}")
    
    class Args:
        def __init__(self):
            self.config = config_name
            self.problem = None
            self.case_name = None
            self.method = None
            self.epochs = None
            self.save_flag = True
            self.gen_flag = False
            self.debug_flag = True
            self.random_seed = 42
            
    args = Args()
    params = Param(config_dict, args)
    params.physics = config_dict["physics"]
    
    # 1. Initialize Dataset
    print("Initializing Dataset...")
    try:
        dataset = WindowDataset(params)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return
    
    # 2. Initialize Model
    print("Initializing ConvFormer Model...")
    model = ConvFormer(params)
    
    # Build Model
    T_win = params.architecture["window_size"]
    Nx = dataset.Nx
    dummy_input = tf.zeros((1, T_win, Nx, 3))
    _ = model(dummy_input)
    model.summary()
    
    # 3. Initialize Trainer (Pass output_dir)
    print("Initializing Trainer...")
    try:
        tf.config.experimental.reset_memory_stats('GPU:0')
    except:
        pass
        
    trainer = Seq2SeqTrainer(model, params, dataset, output_dir=output_dir)
    
    # 4. Start Training
    trainer.train()
    
    print("Training sequence completed.")

if __name__ == "__main__":
    main()