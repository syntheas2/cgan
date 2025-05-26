"""CLI."""

import argparse
from pathlib import Path
from cgan.data import read_csv, write_tsv
from cgan.synthesizers.ctgan import CTGAN
from pythelpers.logger.logger import start_logging2
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
    args.gmm_dir = base_dir / '../output/features4ausw4linearsvc_train/gmm'
    args.output = Path(args.output_dir) / 'features4ausw4linearsvc_train_ctgan.csv'
    args.output_model = Path(args.output_dir) / 'model_final.pt'
    args.output_model_best = Path(args.output_dir) / 'model_{}.pt'
    args.ckpt = Path(args.output_dir) / 'model.pt'
    args.logs = Path(args.output_dir) / 'logs'
    args.tensor_logs = Path(args.output_dir) / 'tensor_logs'
    args.gmm_path = str(Path(args.gmm_dir) / 'gmm_model.pkl')
    args.vae_path = str(Path(args.output_dir) / 'train_z.npy')
    args.vaedecoderpath = str(Path(args.output_dir) / 'decoder.pt')

    # model hyperparameters
    args.discriminator_lr = 2e-4
    args.generator_lr = 3e-4
    args.epochs = 2000
    args.batch_size = 256
    args.pac = 1
    args.discriminator_dim = '128,128'
    args.generator_dim = '256,256'

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    run_descr = f"""
    CTGAN training run with the following parameters:
    - epochs: {args.epochs}
    - batch_size: {args.batch_size}
    - generator_lr: {args.generator_lr}
    - discriminator_lr: {args.discriminator_lr}
    - generator_dim: {args.generator_dim}
    - discriminator_dim: {args.discriminator_dim}
    - embedding_dim: {args.embedding_dim}
    - generator_decay: {args.generator_decay}
    - discriminator_decay: {args.discriminator_decay}"""

    with start_logging2('igan strat 1 only vae', run_descr, logs_dirpath=args.logs):
        data, discrete_columns = read_csv(args.data, args.metadata, args.header, args.discrete, ['id', 'combined_tks'])
        
        # only impact as cond
        discrete_condcolumns = []

        if args.load:
            model = CTGAN.load(args.load)
        else:
            generator_dim = [int(x) for x in args.generator_dim.split(',')]
            discriminator_dim = [int(x) for x in args.discriminator_dim.split(',')]
            model = CTGAN(
                embedding_dim=args.embedding_dim,
                generator_dim=generator_dim,
                discriminator_dim=discriminator_dim,
                generator_lr=args.generator_lr,
                generator_decay=args.generator_decay,
                discriminator_lr=args.discriminator_lr,
                discriminator_decay=args.discriminator_decay,
                batch_size=args.batch_size,
                epochs=args.epochs,
                verbose=True,
                tensor_logsdir=args.tensor_logs,
                val_path=args.val_path,
                pac=args.pac,
                model_output_path=args.output_model_best,
                gmm_path=args.gmm_path,
            )
        model.fit(train_data=None, save_dir=args.output_dir, load_checkpoint=args.ckpt, args=args)

        if args.output_model is not None:
            model.save(args.output_model)


if __name__ == '__main__':
    main()
