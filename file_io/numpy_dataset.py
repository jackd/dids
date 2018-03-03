import os
import numpy as np
from file_dataset import FileDataset


class NumpyDataset(FileDataset):

    def __getitem__(self, key):
        return np.load(self.path(key))

    def __contains__(self, key):
        return os.path.isfile(self.path(key))

    def __setitem__(self, key, value):
        self._assert_writable('Cannot set item of unwritable dataset')
        if not isinstance(value, np.ndarray):
            raise TypeError('value must be a numpy array.')
        np.save(self._path(key), value)

    def __delitem__(self, key):
        self._assert_writable('Cannot delete item from unwritable dataset')
        os.remove(self.path(key))
