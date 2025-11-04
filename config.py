"""
Configure your parameters
"""
import os
# Seeding
SEED = 123
DEVICE='cuda:1'

# Training Parameters
MAX_EPOCHS = 1000
VAL_INTERVAL = 1
BATCH_SIZE = 50
LEARNING_RATE = 1e-1
NUM_WORKERS = 8
MODEL_TYPE = 'sfcne'
MODEL_NAME = f'final_model_b{BATCH_SIZE}_e{MAX_EPOCHS}.pt'

# Loss Parameters
CONTRASTIVE_TEMPERATURE = 0.05

# Early Stopping
PATIENCE = 10
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5

# Data
COHORT_SSL = 'multi-dis'
COHORT_EXTRACT = 'id1000'
IMG_SIZE = 96

# ============================================
# DIMENSIONALITY REDUCTION SETTINGS
# ============================================

# Choose reduction method: 'umap', 'tsne', or 'pca'
REDUCTION_METHOD = 'tsne'  # Options: 'umap', 'tsne', 'pca'
# UMAP-specific parameters
N_NEIGHBORS = 10        # Number of neighbors (UMAP only)
MIN_DIST = 1.0         # Minimum distance (UMAP only)
# t-SNE-specific parameters
PERPLEXITY = 30        # Perplexity (t-SNE only, typically 5-50)
# General parameters
RANDOM_STATE = 3       # Random seed for reproducibility

# ============================================
# CONFIGURE PATHS
# ============================================

# File Paths (CHANGE THESE TO YOUR PATHS)
JSON_PATH = f'../data/ssl_data/ssl-{COHORT_SSL}.json'
LOG_DIR = f"../logs/ssl/{MODEL_TYPE}/{COHORT_SSL}/{COHORT_SSL}{IMG_SIZE}/"
MODEL_DIR = f"../models/ssl/{MODEL_TYPE}/{COHORT_SSL}/{COHORT_SSL}{IMG_SIZE}/"
IMAGES_EXT_DIR = f'../images/{COHORT_EXTRACT}/npy96/'
FEATURES_EXT_DIR = f"../features/{COHORT_EXTRACT}/{MODEL_TYPE}/"
VIZ_DIR = f"../representations/{COHORT_EXTRACT}/{MODEL_TYPE}/ssl-{COHORT_SSL}/{MODEL_NAME}/{REDUCTION_METHOD}/"
DATA_PATH = f'../data/{COHORT_EXTRACT}/demographics.csv'


# ============================================
# VIZUALIZATION SETTINGS
# ============================================
POINT_SIZE = 200
TRANSPARENCY = 0.7
FONTSIZE_MAX = 30
FONTSIZE_MIN = 20