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
from pythelpers.ml.data_transformer import DataTransformer
from pipelines.train_cgan_args import CGANArgs
from cgan.data_sampler import DataSampler
# import vae_syntheas
# at first try train tabsyn

ReturnType = Tuple[    
    Annotated[Any, "data_transformer"],
    Annotated[Any, "data_sampler"],
    Annotated[Any, "np_val_transformed"],
    ]

@step
def prepare_data_step(config: CGANArgs, discrete_columns, discrete_condcolumns, df_train, df_val) -> ReturnType:

    validate_discrete_columns(df_train, discrete_columns)
    validate_null_data(df_train, discrete_columns)

    data_transformer = DataTransformer()
    data_transformer.fit(df_train, discrete_columns)

    np_train_transformed = data_transformer.transform(df_train)
    np_val_transformed = data_transformer.transform(df_val)

    data_sampler = DataSampler(
        np_train_transformed, 
        data_transformer.output_info_list, 
        config.log_frequency,
        discrete_columns=discrete_columns,
        cond_column_names=discrete_condcolumns
    )

    return data_transformer, data_sampler, np_val_transformed