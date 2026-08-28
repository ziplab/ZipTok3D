from torch.utils.data import Dataset
import h5py


class HDF5Dataset(Dataset):

    def __init__(self, file_path):
        super().__init__()

        self._file_path = file_path
        self._file = None

    @property
    def file(self):
        if self._file is None:
            self._file = h5py.File(self._file_path, 'r', swmr=True)

        return self._file

    def close_file(self):
        if self._file is None:
            return

        self._file.close()
        self._file = None