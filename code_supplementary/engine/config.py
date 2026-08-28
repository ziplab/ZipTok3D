from typing import Union, Callable
import inspect
import importlib
import functools

from os import path

from omegaconf import OmegaConf, DictConfig

RESERVED_KEY_BASE = '_base_'
RESERVED_KEY_TARGET = '_target_'
RESERVED_KEY_OVERWRITE = '_overwrite_'

__all__ = ['instantiate', 'load_config']


def instantiate(cfg: Union[DictConfig, dict],
                target: Callable = None,
                remove_unknown_params: bool = False,
                allow_unknown_params: bool = False,
                as_callable: bool = False,
                **overrides):
    """
    A simple replacement of hydra.utils.instantiate method, with some extra features:
    - explicitly specifying target when instantiating if needed
    - explicitly overriding some arguments when instantiating
    - setting remove_known_params=True to automatically remove unknown arguments from the cfg
    - returning factory instead of the instance if as_callable=True
    :param cfg:
    :param target:
    :param remove_unknown_params:
    :param allow_unknown_params:
    :param as_callable:
    :param overrides:
    :return:
    """
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)

    current_key = cfg._key()
    if target is None:
        target_path = cfg.get(RESERVED_KEY_TARGET, None)
        assert isinstance(target_path, str), f'{RESERVED_KEY_TARGET} or target should be specified (cfg.{current_key})'
        target = _import_target(target_path)

    # to resolve interpolated values of OmegaConf
    parent_cfg = cfg._get_parent()
    params = cfg.copy()
    params._set_parent(parent_cfg)

    ### dict config to dict
    params_dict = OmegaConf.to_container(params, resolve=True)
    params_dict = _merge_dict(params_dict, overrides)

    _delete_reserved_keys_(params_dict)
    unknown_keys = _get_unknown_param_keys(target, params_dict)
    if remove_unknown_params:
        params_dict = _filter_unknown_params(params_dict, unknown_keys)
    elif len(unknown_keys) > 0 and not allow_unknown_params:
        raise Exception(f'({target.__name__}) unknown keys: {",".join(unknown_keys)}')

    if as_callable:
        return functools.partial(target, **params_dict)

    try:
        return target(**params_dict)
    except Exception as e:
        error_name = e.__class__.__name__ if e is not None else ''
        message = f'({target.__name__}) {error_name}: {e.args[0]}'
        raise Exception(message)


def _merge_dict(src, target, copy=True):
    if copy:
        src = src.copy()
    for key, value in target.items():
        src[key] = value
        pass

    return src


def _delete_reserved_keys_(params_dict):
    keys = list(params_dict.keys())
    for key in keys:
        if key in [RESERVED_KEY_BASE, RESERVED_KEY_TARGET, RESERVED_KEY_OVERWRITE]:
            del params_dict[key]
        pass
    pass


def _get_unknown_param_keys(fn, params_dict):
    spec = inspect.getfullargspec(fn)
    if spec.varkw is not None:
        return []
    args = spec.args
    if params_dict is None:
        return args
    if len(args) == 0:
        return []
    if args[0] == 'self':
        args = args[1:]

    unknown_keys = []
    for key in params_dict.keys():
        if key not in args:
            unknown_keys.append(key)

    return unknown_keys


def _filter_unknown_params(params_dict, unknown_keys):
    if params_dict is None:
        return {}

    filtered_params = {}
    for key in params_dict.keys():
        if key not in unknown_keys:
            filtered_params[key] = params_dict[key]

    return filtered_params


def _import_target(class_path):
    packages = class_path.split('.')
    class_name = packages[-1]
    packages = '.'.join(packages[:-1])

    try:
        module = importlib.import_module(packages)
    except Exception as e:
        error_name = e.__class__.__name__ if e is not None else ''
        message = f'({class_name}) {error_name}: {e.args[0]}'
        raise Exception(message)

    if not hasattr(module, class_name):
        raise Exception(f'({class_name}) failed to find {class_name} from "{packages}"')

    return getattr(module, class_name)


def load_config(config_path: str, root_config_dir: str = None):
    assert _file_is_yaml(config_path), 'config file must be a yaml file'

    if root_config_dir is None:
        root_config_dir = path.dirname(config_path)
    return _load_config_file_with_base(config_path, root_config_dir)


def _load_config_file_with_base(config_path: str, root_config_dir: str):
    raw_cfg = OmegaConf.load(config_path)
    cfg = raw_cfg.copy()

    candidate_keys = []
    if _has_base(cfg):
        candidate_keys.append('')
    candidate_keys.extend([x for x in raw_cfg.keys() if _has_base(cfg[x])])
    for key in candidate_keys:
        is_root = key == ''
        value = cfg if is_root else cfg[key]

        target = value[RESERVED_KEY_BASE]
        sub_config_path = _search_base_config_path(target, path.dirname(config_path), root_config_dir)
        if sub_config_path is None:
            raise FileNotFoundError(f'fail to find sub config: {target}')

        base_cfg = _load_config_file_with_base(sub_config_path, root_config_dir)
        if (RESERVED_KEY_OVERWRITE in value):
            merged = _merge_with_overwrite(base_cfg, value)
            del cfg[key][RESERVED_KEY_OVERWRITE]
        else:
            merged = OmegaConf.merge(base_cfg, value)

        if is_root:
            cfg = merged
            del cfg[RESERVED_KEY_BASE]
        else:
            cfg[key] = merged
            del cfg[key][RESERVED_KEY_BASE]

    # for key, value in raw_cfg.items():
    #     if isinstance(value, DictConfig) and (RESERVED_KEY_BASE in value):
    #         target = value[RESERVED_KEY_BASE]
    #         sub_config_path = _search_base_config_path(target, path.dirname(config_path), root_config_dir)
    #         if sub_config_path is None:
    #             raise FileNotFoundError(f'fail to find sub config: {target}')
    #
    #         base_cfg = _load_config_file_with_base(sub_config_path, root_config_dir)
    #         if (RESERVED_KEY_OVERWRITE in value):
    #             merged = _merge_with_overwrite(base_cfg, value)
    #             del cfg[key][RESERVED_KEY_OVERWRITE]
    #         else:
    #             merged = OmegaConf.merge(base_cfg, value)
    #
    #         cfg[key] = merged
    #         del cfg[key][RESERVED_KEY_BASE]

    return cfg


def _has_base(cfg):
    return isinstance(cfg, DictConfig) and (RESERVED_KEY_BASE in cfg)


def _merge_with_overwrite(base_cfg, new_cfg):
    base_cfg = base_cfg.copy()
    for key, value in new_cfg.items():
        if key in base_cfg:
            base_cfg[key] = value
    return base_cfg


def _search_base_config_path(target: str, config_dir: str, root_config_dir: str) -> Union[str, None]:
    """
    Searches target base config file from the current config directory to the root config directory.
    :param config_dir:
    :param root_config_dir:
    :return: str
    """
    current_config_dir = config_dir
    while True:
        base_config_path = path.join(current_config_dir, target)
        if not _file_is_yaml(base_config_path):
            base_config_path += '.yaml'
        if path.exists(base_config_path):
            return base_config_path

        if current_config_dir == root_config_dir:
            break
        current_config_dir = path.dirname(current_config_dir)

    return None


def _file_is_yaml(file_path):
    return file_path.endswith('.yaml') or file_path.endswith('.yml')
