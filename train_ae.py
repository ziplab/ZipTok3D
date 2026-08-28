from os import path
from argparse import ArgumentParser

from omegaconf import OmegaConf
import lightning.pytorch as pl

import engine
from cod.utils.training import load_model_weights_from_checkpoint

parser = ArgumentParser('ZipTok3D training entry point')
parser.add_argument('config', type=str, help='Path to the config file')
parser.add_argument('--name', '-n', type=str, default=None, help='Name of the experiment')
parser.add_argument('--debug', '-d', action='store_true', default=False, help='debug mode (for sanity check)')
parser.add_argument('--resume', '-r', type=str, default=None, help='Path to the checkpoint to resume from')
parser.add_argument(
    '--init-checkpoint', type=str, default=None,
    help='Load model weights from a checkpoint without restoring optimizer state or epoch',
)
parser.add_argument(
    '--set', dest='overrides', action='append', default=[], metavar='KEY=VALUE',
    help='Override a config value; repeat for multiple values',
)
parser.add_argument('--gpus', '-g', default='-1',
                    help='GPU to use (num. GPU or gpu ids, follow pytorch-lightning convention). e.g., "-1" (all), "2" (2 GPU), "0,1" (GPU id 0, 1), "[0]" (GPU id 0)')


def main():
    args = parser.parse_args()
    if args.resume is not None and args.init_checkpoint is not None:
        parser.error('--resume and --init-checkpoint are mutually exclusive')
    cfg = engine.load_config(args.config)
    if args.overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))
    pl.seed_everything(cfg.get('seed', 123456))
    if args.debug:
        args.name = 'debugging'
    engine.create_experiment_context(cfg.get('output_dir', None), args.name)

    with open(path.join(engine.to_experiment_dir('config.yaml')), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    with open(path.join(engine.to_experiment_dir('model.yaml')), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    model = engine.instantiate(cfg.model, allow_unknown_params=True)
    if args.init_checkpoint is not None and not args.debug:
        state_dict = load_model_weights_from_checkpoint(args.init_checkpoint)
        model.load_state_dict(state_dict, strict=True)
        print(f'Initialized model weights from {args.init_checkpoint}.')
    dm = engine.instantiate(cfg.data)
    solver = engine.instantiate(cfg.solver, dm=dm, model=model)
    if args.debug:
        solver.enable_debug()

    resume_ckpt_path = None
    if (args.resume is not None) and not args.debug:
        resume_ckpt_path = args.resume
        print(f'Resuming from {resume_ckpt_path}.')

    trainer: pl.Trainer = engine.prepare_trainer(cfg, gpus=args.gpus, debug=args.debug)
    trainer.fit(model=solver, datamodule=dm, ckpt_path=resume_ckpt_path)


if __name__ == '__main__':
    main()
