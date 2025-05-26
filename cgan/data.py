"""Data loading."""

import json

import numpy as np
import pandas as pd
from pathlib import Path
import torch
from cgan.utils import transform_input_data


def read_csv(csv_filename, meta_filename=None, header=True, discrete=None, columns_exclude=[]):
    """Read a csv file."""
    data = pd.read_csv(csv_filename, header='infer' if header else None)

    for col in columns_exclude:
        if col in data.columns:
            data.drop(columns=col, inplace=True)

    if meta_filename:
        with open(meta_filename) as meta_file:
            metadata = json.load(meta_file)

        discrete_columns = [
            column['name'] for column in metadata['columns'] if column['type'] != 'continuous' and column['name'] not in columns_exclude
        ]

    elif discrete:
        discrete_columns = discrete.split(',')
        if not header:
            discrete_columns = [int(i) for i in discrete_columns]

    else:
        discrete_columns = []

    return data, discrete_columns

def read_csv2(csv_filename, header=True, columns_exclude=[]):
    """Read a csv file."""
    data = pd.read_csv(csv_filename, header='infer' if header else None)

    for col in columns_exclude:
        if col in data.columns:
            data.drop(columns=col, inplace=True)

    return data


def get_train_z(args):
    embedding_save_path = Path(args.vae_path)
    train_z = torch.tensor(np.load(embedding_save_path)).float()

    train_z = train_z[:, 1:, :]
    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim
    
    train_z = train_z.view(B, in_dim)

    # normalisierung um 0 -> für stabilere gradienten (mehr in readme)
    mean, std = train_z.mean(0), train_z.std(0)
    train_z = (train_z - mean) / 2
    hid_dim = train_z.shape[1]

    return train_z, hid_dim

def get_sample_conditions(train_df, test_df):
    _, _, categories, d_numerical, num_inverse, cat_inverse, original_columns, column_metadata = transform_input_data(train_df, test_df, inverse = True)



    return num_inverse, cat_inverse, original_columns, column_metadata



def write_tsv(data, meta, output_filename):
    """Write to a tsv file."""
    with open(output_filename, 'w') as f:
        for row in data:
            for idx, col in enumerate(row):
                if idx in meta['continuous_columns']:
                    print(col, end=' ', file=f)
                else:
                    assert idx in meta['discrete_columns']
                    print(meta['column_info'][idx][int(col)], end=' ', file=f)

            print(file=f)

def write_csv(data, meta, output_filename):
    """
    Write data to a CSV file with proper formatting.
    
    Args:
        data: The data to write, as a list of rows
        meta: Metadata containing column types and information
        output_filename: Path to the output CSV file
    """
    import csv
    
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write each row
        for row in data:
            formatted_row = []
            for idx, col in enumerate(row):
                if idx in meta['continuous_columns']:
                    # For continuous columns, just use the value as is
                    formatted_row.append(col)
                else:
                    # For discrete columns, look up the categorical value
                    assert idx in meta['discrete_columns']
                    formatted_row.append(meta['column_info'][idx][int(col)])
            
            writer.writerow(formatted_row)
    
    print(f"CSV file successfully written to {output_filename}")
