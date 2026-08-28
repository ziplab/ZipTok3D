import os
from os import path
import time

__all__ = ['create_experiment_context', 'to_experiment_dir', 'get_experiment_name', 'set_context_from_existing']


class _ExperimentContext:
    def __init__(self, output_root_dir: str = None, name: str = None):
        self.output_root_dir = output_root_dir
        self.name = name

    def from_new_experiment(self, output_root_dir: str = None, name: str = None):
        if output_root_dir is None:
            output_root_dir = 'logs'
        if name is None:
            name = time.strftime('experiment-%Y-%m-%d_%H-%M-%S')
        self.output_root_dir = output_root_dir
        self.name = name
        os.makedirs(self.to_experiment_dir(), exist_ok=True)

    def from_existing_experiment(self, full_output_dir):
        normalized = path.normpath(full_output_dir)
        self.output_root_dir = path.dirname(normalized) or '.'
        self.name = path.basename(normalized)

    def to_experiment_dir(self, *target_dirs):
        self._assert_context_is_set()
        return path.join(self.output_root_dir, self.name, *target_dirs)

    def get_experiment_name(self):
        self._assert_context_is_set()
        return self.name

    def _assert_context_is_set(self):
        is_set = (self.output_root_dir is not None) and (self.name is not None)
        assert is_set, 'Experiment context is not set. Call set_experiment_context first'


_context = _ExperimentContext()
create_experiment_context = _context.from_new_experiment
set_context_from_existing = _context.from_existing_experiment
to_experiment_dir = _context.to_experiment_dir
get_experiment_name = _context.get_experiment_name
