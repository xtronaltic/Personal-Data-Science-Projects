import os
import random
import numpy as np
import torch

def set_global_seeds(seed: int = 42) -> None:
    """Set global seeds for reproducibility across random, numpy, and torch."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Try to set tensorflow seed if available (for TimesFM if it uses it)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    print(f"Global random seed set to {seed}")
