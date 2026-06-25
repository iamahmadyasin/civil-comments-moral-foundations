import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
DATA_DIR    = os.path.join(BASE_DIR, "data")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")

# Hugging Face
HF_TOKEN    = "hf_xxxxxxxxxxxxxxxxxxxxxxxx"  # ← paste your token here

# Dataset
DATASET_NAME   = "google/civil_comments"
SAMPLE_SIZE    = 10_000

# File names
MF_SCORES_FILE = os.path.join(DATA_DIR, "civil_comments_with_mf_scores.csv")
