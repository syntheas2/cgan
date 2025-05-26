from pydantic import BaseModel
from typing import Optional, Union
from pathlib import Path
import torch

class CGANArgs(BaseModel):
    # Data and Execution
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4

    # Data paths and metadata
    data: Union[str, Path] = ""
    val_path: Union[str, Path] = ""
    metadata: Union[str, Path] = ""
    output_dir: Union[str, Path] = ""
    gmm_dir: Union[str, Path] = ""
    output: Union[str, Path] = ""
    output_model: Union[str, Path] = ""
    output_model_best: Union[str, Path] = ""
    ckpt: Union[str, Path] = ""
    logs: Union[str, Path] = ""
    tensor_logs: Union[str, Path] = ""
    gmm_path: Union[str, Path] = ""
    vae_path: Union[str, Path] = ""
    vaedecoderpath: Union[str, Path] = ""

    # Data format
    tsv: bool = False
    header: bool = True
    discrete: Optional[str] = None  # comma separated list of discrete columns
    sample_condition_column: Optional[str] = None
    sample_condition_column_value: Optional[str] = None

    # Model Hyperparameters
    embedding_dim: int = 128
    generator_dim: str = '256,256'
    discriminator_dim: str = '128,128'
    pac: int = 1

    # Training Hyperparameters
    epochs: int = 2000
    batch_size: int = 256
    generator_lr: float = 3e-4
    discriminator_lr: float = 2e-4
    generator_decay: float = 1e-6
    discriminator_decay: float = 0.0
    num_samples: Optional[int] = None
    save: Optional[str] = None

    # Diffusion/MLP
    dim_token: int = 1024  # for MLPDiffusion if used
    # VAE
    vae_token_dim: int = 4  # for MLPDiffusion if used

    # Scheduler and early stopping
    scheduler_patience: int = 20
    scheduler_factor: float = 0.9
    early_stopping_patience: int = 500

    # MLflow
    mlflow_experiment_name: str = "CGAN_Training_Experiment"
    load_ckp_from_run_id: str = ""
    load_from_checkpoint: bool = True
    max_checkpoints_to_keep: int = 5
    checkpoint_save_interval: int = 20
    manual_checkpoint_subdir: str = "manual_model_checkpoints" # Define consistently
    manual_bestmodel_subdir: str = "manual_model_best" # Define consistently
    max_bestmodel_to_keep: int = 2 # Define consistently
    bestmodels_runid: str | None = None