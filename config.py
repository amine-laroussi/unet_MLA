"""
Configuration paths for the MLA project.
Paths are computed relative to the project root.
"""
import os

# Project root (parent of this file)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATASET_NORMALIZER_DIR = os.path.join(PROJECT_ROOT, "dataset_normalizer")
UNET_MODEL_DIR         = os.path.join(PROJECT_ROOT, "unet_model")
NOTEBOOKS_DIR          = os.path.join(PROJECT_ROOT, "notebooks")
SAVED_MODELS_DIR       = os.path.join(PROJECT_ROOT, "saved_models")

# Raw dataset paths
RAW_ISBI_IMG_DATASET_DIR          = os.path.join(PROJECT_ROOT, "bdd", "non_normalized", "ISBI", "Img")
RAW_ISBI_GT_DATASET_DIR           = os.path.join(PROJECT_ROOT, "bdd", "non_normalized", "ISBI", "GT")

# Normalized dataset paths
ISBI_DATASET_DIR           = os.path.join(PROJECT_ROOT, "bdd", "normalized", "ISBI")

# Model save
UNET_ISBI              = "unet_isbi.pth"