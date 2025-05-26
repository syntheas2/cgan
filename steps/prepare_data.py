import __init__ # noqa: F401
from typing import Annotated, Tuple, Dict, Any
import pandas as pd
from zenml import step
from typing import Annotated
from zenml.client import Client
import torch
import torch.nn as nn
from cgan.data import get_sample_conditions
from cgan.utils_evaluation import transform_val_data
from cgan.utils import validate_discrete_columns, validate_null_data
from cgan.data_transformer import DataTransformer
# import vae_syntheas
# at first try train tabsyn

ReturnType = Tuple[    
    Annotated[Any, "X_val"],
    Annotated[Any, "y_val"],
    Annotated[Any, "data_transformer"],
    ]

@step
def prepare_data_step(discrete_columns, df_train, df_val) -> ReturnType:
    X_val, y_val = transform_val_data(df_val)

    validate_discrete_columns(df_train, discrete_columns)
    validate_null_data(df_train, discrete_columns)

    data_transformer = DataTransformer()
    data_transformer.fit(df_train, discrete_columns)

    return X_val, y_val, data_transformer