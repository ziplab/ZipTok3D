from os import path
from argparse import ArgumentParser

import lightning.pytorch as pl

import engine
from cod.solvers.recon_eval import ReconEvaluator

DEFAULT_OUTPUT_DIR = 'results/'

parser = ArgumentParser('Autoencoder evaluation')
parser.add_argument('model_dir', type=str, help='path to the saved weights dir')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='checkpoint to evaluate; defaults to the best checkpoint in model_dir')
parser.add_argument('--save_dir', '-sd', type=str, default=None, help='path to the output dir')
parser.add_argument('--data', '-d', type=str, default=None, help='name of the data config (e.g., shapenet)')
parser.add_argument('--eval', '-e', type=str, default=None, help='name of the evaluator config')
parser.add_argument('--seed', '-s', type=int, default=123456, help='evaluation random seed')
parser.add_argument('--tokens', type=int, default=None,
                    help='exact retained stage-1 prefix length (no suffix padding)')
parser.add_argument('--loops', type=int, default=None,
                    help='number of shared decoder refinement passes')
parser.add_argument('--split', choices=('val', 'test'), default=None,
                    help='physical dataset split; defaults to the configured evaluation split')
parser.add_argument('--allow-invalid-meshes', action='store_true',
                    help='diagnostic only; paper metrics require a valid mesh for every object')
parser.add_argument('--gpus', '-g', default='[0]',
                    help='GPU to use (num. GPU or gpu ids, follow pytorch-lightning convention). e.g., "-1" (all), "2" (2 GPU), "0,1" (GPU id 0, 1), "[0]" (GPU id 0)')


def main():
    args = parser.parse_args()
    pl.seed_everything(args.seed)

    cfg = engine.load_config(path.join(args.model_dir, 'config.yaml'))
    checkpoint_path = args.checkpoint or engine.find_best_checkpoint_path(
        path.join(args.model_dir, 'checkpoints')
    )
    if checkpoint_path is None or not path.isfile(checkpoint_path):
        raise FileNotFoundError(
            'no evaluable checkpoint found; pass --checkpoint explicitly'
        )
    output_dir = args.model_dir

    model_name = path.basename(path.normpath(args.model_dir))
    if args.save_dir is not None:
        output_dir = path.join(args.save_dir, model_name)
    engine.set_context_from_existing(output_dir)

    model = engine.instantiate(cfg.model)
    if args.data is not None:
        data_cfg = engine.load_config(path.join('config/data', f'{args.data}.yaml'))
    elif 'data' in cfg:
        data_cfg = cfg.data
    else:
        raise Exception('data config should be specified either from command line or config file')
    dm = engine.instantiate(data_cfg)
    split = args.split or dm.evaluation_split

    tokens = args.tokens if args.tokens is not None else getattr(model, 'num_latents', 'full')
    decoder = getattr(model, 'decoder', None)
    loops = args.loops if args.loops is not None else getattr(decoder, 'num_loops', 1)
    setting_name = f'{split}_k{tokens}_l{loops}'

    eval_cfg = {}
    if args.eval is not None and path.exists(args.eval):
        eval_cfg = engine.load_config(args.eval)
    evaluator = engine.instantiate(
        eval_cfg,
        ReconEvaluator,
        dm=dm,
        model=model,
        active_num_latents=args.tokens,
        num_decode_loops=args.loops,
        output_subdir=path.join('outputs', 'recon', setting_name),
        metric_seed=args.seed,
        allow_invalid_meshes=args.allow_invalid_meshes,
    )
    evaluator.restore_checkpoint(checkpoint_path)

    gpus = engine.parse_gpus_str(args.gpus)
    trainer = pl.Trainer(devices=gpus)
    dataloader = dm.eval_dataloader(split)
    trainer.test(model=evaluator, dataloaders=dataloader)

    ### CD metric
    metric_dataset = dm.get_dataset(split)
    metric_dataset.use_queries = False
    metric_dataset.use_full_surface = True
    evaluator.measure_cd(metric_dataset)


if __name__ == '__main__':
    main()
