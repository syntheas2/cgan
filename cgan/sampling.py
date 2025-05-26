"""CLI."""

import argparse
from pathlib import Path
from cgan.data import read_csv, get_sample_conditions
from cgan.synthesizers.ctgan import CTGAN
from cgan.utils import split_num_cat_target, recover_data
from pythelpers.logger.logger import start_logging, stop_logging
import os

base_dir = Path(__file__).parent

def _parse_args():
    parser = argparse.ArgumentParser(description='CTGAN Command Line Interface')
    parser.add_argument('-e', '--epochs', default=300, type=int, help='Number of training epochs')
    parser.add_argument(
        '-t', '--tsv', action='store_true', help='Load data in TSV format instead of CSV'
    )
    parser.add_argument(
        '--no-header',
        dest='header',
        action='store_false',
        help='The CSV file has no header. Discrete columns will be indices.',
    )

    parser.add_argument('-m', '--metadata', help='Path to the metadata')
    parser.add_argument(
        '-d', '--discrete', help='Comma separated list of discrete columns without whitespaces.'
    )
    parser.add_argument(
        '-n',
        '--num-samples',
        type=int,
        help='Number of rows to sample. Defaults to the training data size',
    )

    parser.add_argument(
        '--generator_lr', type=float, default=2e-4, help='Learning rate for the generator.'
    )
    parser.add_argument(
        '--discriminator_lr', type=float, default=2e-4, help='Learning rate for the discriminator.'
    )

    parser.add_argument(
        '--generator_decay', type=float, default=1e-6, help='Weight decay for the generator.'
    )
    parser.add_argument(
        '--discriminator_decay', type=float, default=0, help='Weight decay for the discriminator.'
    )

    parser.add_argument(
        '--embedding_dim', type=int, default=128, help='Dimension of input z to the generator.'
    )
    parser.add_argument(
        '--generator_dim',
        type=str,
        default='256,256',
        help='Dimension of each generator layer. Comma separated integers with no whitespaces.',
    )
    parser.add_argument(
        '--discriminator_dim',
        type=str,
        default='256,256',
        help='Dimension of each discriminator layer. Comma separated integers with no whitespaces.',
    )

    parser.add_argument(
        '--batch_size', type=int, default=500, help='Batch size. Must be an even number.'
    )
    parser.add_argument(
        '--save', default=None, type=str, help='A filename to save the trained synthesizer.'
    )
    parser.add_argument(
        '--load', default=None, type=str, help='A filename to load a trained synthesizer.'
    )

    parser.add_argument(
        '--sample_condition_column', default=None, type=str, help='Select a discrete column name.'
    )
    parser.add_argument(
        '--sample_condition_column_value',
        default=None,
        type=str,
        help='Specify the value of the selected discrete column.',
    )

    parser.add_argument('--data', help='Path to training data')
    parser.add_argument('--output', help='Path of the output file')

    return parser.parse_args()


def main():
    """CLI."""
    args = _parse_args()
    args.data = base_dir / '../input/features4ausw4linearsvc_train.csv'
    args.val_path = str(base_dir / '../input/features4ausw4linearsvc_test.csv')
    args.metadata = base_dir / '../input/features4ausw4linearsvc_train.json'
    args.output_dir = base_dir / '../output/features4ausw4linearsvc_train12'
    args.output = Path(args.output_dir) / 'features4ausw4linearsvc_train_ictgan.csv'
    args.output_model = Path(args.output_dir) / 'model_0.98.pt'
    args.ckpt = Path(args.output_dir) / 'model.pt'
    args.logs = Path(args.output_dir) / 'logs'
    args.vae_path = str(Path(args.output_dir) / 'train_z.npy')
    args.vaedecoderpath = str(Path(args.output_dir) / 'decoder.pt')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger = start_logging('cgan sampling', '', logs_dirpath=args.logs)

    data, discrete_columns = read_csv(args.data, args.metadata, args.header, args.discrete, ['id', 'combined_tks'])

    if args.output_model:
        model = CTGAN.load(args.output_model)
    else:
        raise ValueError('Load model is required for sampling.')

    num_samples = args.num_samples or len(data)

    if args.sample_condition_column is not None:
        assert args.sample_condition_column_value is not None

    train_z, in_dim, mean, pre_decoder, num_inverse, cat_inverse, original_columns, column_metadata = get_sample_conditions(args)

    x_next = model.sample(
        num_samples
    )

    x_next = x_next * 2 + mean.numpy()
    syn_data = x_next
    syn_num, syn_cat_df, syn_target = split_num_cat_target(syn_data, pre_decoder, in_dim, column_metadata, num_inverse, cat_inverse)


    # Wiederherstellung der ursprünglichen Datenstruktur
    target_column = column_metadata.get('target_column')
    numerical_columns = column_metadata['numerical_columns'] 
    generated_data = recover_data(syn_num, syn_cat_df, syn_target, numerical_columns, target_column)

    generated_data.to_csv(args.output, index=False)

    stop_logging(logger)


if __name__ == '__main__':
    main()
