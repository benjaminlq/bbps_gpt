import os

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(MAIN_DIR, "data")
ARTIFACT_DIR = os.path.join(MAIN_DIR, "artifacts")
SEED = 2023