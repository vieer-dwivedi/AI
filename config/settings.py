import torch

# ===============================
# Model Configuration
# ===============================

MODEL_NAME = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

# Automatically choose device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Use fp16 only when GPU is available
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# ===============================
# Generation Settings
# ===============================

MAX_NEW_TOKENS = 256

# Deterministic output (best for agents)
DO_SAMPLE = False

# Only used if DO_SAMPLE = True
TEMPERATURE = 0.7
TOP_P = 0.9

# ===============================
# Agent Settings
# ===============================

MAX_AGENT_STEPS = 5   # Prevent infinite loops
VERBOSE = True        # Print agent thoughts & actions

# ===============================
# Memory Settings (optional)
# ===============================

ENABLE_MEMORY = False
MEMORY_WINDOW = 10    # Last N messages to keep

# ===============================
# Logging / Debug
# ===============================

LOG_LEVEL = "INFO"    # DEBUG | INFO | WARNING | ERROR