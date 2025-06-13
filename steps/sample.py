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
from pipelines.sample_cgan_args import CGANArgs
from cgan.data_sampler import DataSampler
from cgan.main import Generator
from pythelpers.ml.data_transformer import DataTransformer
# import vae_syntheas
# at first try train tabsyn

ReturnType = Annotated[Any, "cgan_df_syn"]


@step
def sample_step(config: CGANArgs, generator: Generator, transformer: DataTransformer, discrete_column_category_prob_flatten, discrete_column_matrix_st, n_categories, orig_columns) -> ReturnType:
            # generate samples
    generated_data_tr = generator.sample(config.n_sample, discrete_column_category_prob_flatten, n_categories, discrete_column_matrix_st, transformer)

    # Wiederherstellung der ursprünglichen Datenstruktur
    generated_data_inv_tr = transformer.inverse_transform(generated_data_tr)
    generated_data_inv_tr = generated_data_inv_tr[orig_columns].copy()



    return generated_data_inv_tr