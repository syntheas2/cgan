import __init__ # noqa: F401
from typing import Annotated, Tuple, Dict, Any
import pandas as pd
from zenml import step
from typing import Annotated
from zenml.client import Client
import torch
import torch.nn as nn
from pythelpers.ml.mlflow import load_best_model
from pipelines.sample_cgan_args import CGANArgs as SampleCGANArgs
from cgan.load_model import load_for_sampling
from syntheas_mltools_mgmt.zenml.data import get_df_val, get_df_train
# import vae_syntheas
# at first try train tabsyn

ReturnType = Tuple[    
    Annotated[pd.DataFrame, "df_train"],
    Annotated[pd.DataFrame, "df_val"],
    Annotated[Any, "discrete_columns"],
    ]

@step
def load_data_step() -> ReturnType:
    df_train = get_df_train()
    try:
        df_train.drop(columns=['id', 'combined_tks'], inplace=True)
    except KeyError:
        # If the columns are not present, we can ignore this error
        pass
    df_val = get_df_val()
    try:
        df_val.drop(columns=['id', 'combined_tks'], inplace=True)
    except KeyError:
        # If the columns are not present, we can ignore this error
        pass

    discrete_columns = set(df_train.columns.to_list())

    return df_train, df_val, discrete_columns

ReturnType2 = Tuple[    
    Annotated[Any, "generator"],
    Annotated[Any, "transformer"],
    Annotated[Any, "discrete_column_category_prob_flatten"],
    Annotated[Any, "discrete_column_matrix_st"],
    Annotated[Any, "n_categories"],
    Annotated[Any, "orig_columns"],
    ]
@step
def load_data4sample_step(args: SampleCGANArgs) -> ReturnType2:
    model_ckp = load_best_model(
        args.bestmodels_runid,
        args.manual_bestmodel_subdir,
        "acc",
        maximize=True,
    )

    df_val = get_df_val()
    df_val.drop(columns=['id', 'combined_tks'], inplace=True)

    generator, transformer, discrete_column_category_prob_flatten, discrete_column_matrix_st, n_categories = load_for_sampling(model_ckp)
    orig_columns = df_val.columns

    return generator, transformer, discrete_column_category_prob_flatten, discrete_column_matrix_st, n_categories, orig_columns