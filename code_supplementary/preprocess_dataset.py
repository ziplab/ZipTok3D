from os import path

from argparse import ArgumentParser

import engine
from cod.data.base import BaseDataModule

DEFAULT_OUTPUT_DIR = 'results/'

parser = ArgumentParser('ZipTok3D dataset preprocessing')
parser.add_argument('--data', '-d', type=str, default=None, help='name of the data config (e.g., shapenet)')


def main():
    args = parser.parse_args()
    if args.data is None:
        raise ValueError('Data config must be specified with --data')

    data_cfg = engine.load_config(path.join('config/data', f'{args.data}.yaml'))
    dm: BaseDataModule = engine.instantiate(data_cfg)
    dm.preprocess()


if __name__ == '__main__':
    main()
