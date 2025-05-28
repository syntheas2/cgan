import __init__ # noqa: F401
from typing import Annotated, Tuple, Dict, Any
import pandas as pd
from zenml import step
from typing import Annotated
from zenml.client import Client
import torch
import torch.nn as nn
# import vae_syntheas
# at first try train tabsyn

ReturnType = Tuple[    
    Annotated[pd.DataFrame, "df_train"],
    Annotated[pd.DataFrame, "df_val"],
    Annotated[Any, "discrete_columns"],
    Annotated[Any, "one_hot_columns"],
    ]

@step
def load_data_step() -> ReturnType:
    artifact = Client().get_artifact_version(
        "b399655a-01d4-47a0-9c1d-92ef258a0023")
    df_train = artifact.load()
    df_train.drop(columns=['id', 'combined_tks'], inplace=True)

    artifact2 = Client().get_artifact_version(
        "c5f0a53c-939b-4ddd-a68f-e58756dc864a")
    df_val = artifact2.load()
    df_val.drop(columns=['id', 'combined_tks'], inplace=True)

    discrete_columns = set(['impact'])
    one_hot_columns = set(df_train.columns.to_list()) - discrete_columns

    return df_train, df_val, discrete_columns, one_hot_columns