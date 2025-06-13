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

    # sample Hyperparameters
    n_sample: int = 1000000


    # MLflow
    mlflow_experiment_name: str = "CGAN_sample"
    mlflow_run_name: str = "init"
    mlflow_run_desc: str = """"""
    manual_bestmodel_subdir: str = "manual_model_best" # Define consistently
    bestmodels_runid: str | None = "c91dd10a7035465c97b8cff88624057f"