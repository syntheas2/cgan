from pydantic import BaseModel
from typing import Optional, Union
from pathlib import Path
import torch

class CGANArgs(BaseModel):
    # Data and Execution
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4

    # Data paths and metadata
    metadata: Union[str, Path] = ""
    log_frequency: bool = True  # Whether to log training progress

    # Data format
    tsv: bool = False
    header: bool = True
    discrete: Optional[str] = None  # comma separated list of discrete columns
    sample_condition_column: Optional[str] = None
    sample_condition_column_value: Optional[str] = None

    # Model Hyperparameters
    embedding_dim: int = 128
    generator_dim: str = '256,256'
    discriminator_dim: str = '256,256'
    pac: int = 1

    # Training Hyperparameters
    epochs: int = 2000
    batch_size: int = 256
    generator_lr: float = 0.002
    discriminator_lr: float = 0.001
    generator_decay: float = 2e-6 # L2 Regularization
    discriminator_decay: float = 1e-6 # L2 Regularization
    num_samples: Optional[int] = None
    save: Optional[str] = None

    # Scheduler and early stopping
    scheduler_patience: int = 20
    scheduler_factor: float = 0.9
    early_stopping_patience: int = 500

    # MLflow
    mlflow_experiment_name: str = "CGAN_Training_Experiment2"
    mlflow_run_name: str = "newparams_5"
    mlflow_run_desc: str = """no pre one hot encoding"""
    load_ckp_from_run_id: str = ""
    load_from_checkpoint: bool = True
    max_checkpoints_to_keep: int = 8
    checkpoint_save_interval: int = 20
    manual_checkpoint_subdir: str = "manual_model_checkpoints" # Define consistently
    manual_bestmodel_subdir: str = "manual_model_best" # Define consistently
    max_bestmodel_to_keep: int = 4 # Define consistently
    bestmodels_runid: str | None = ""