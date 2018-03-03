import os
import numpy as np
import h5py
import dids.core as core
import dids.auto_save as auto_save


class Hdf5Dataset(core.WrappedDictDataset):
    def __init__(self, path, mode='r'):
        self._path = path
        self._mode = mode
        self._base = None

    @property
    def path(self):
        return self._path

    @property
    def is_open(self):
        return self._base is not None

    def open(self):
        if self.is_open:
            raise IOError('Hdf5Dataset already open')
        if self._mode == 'r':
            if not os.path.isfile(self._path):
                raise IOError('No file at %s' % self._path)
        else:
            folder = os.path.dirname(self._path)
            if not os.path.isdir(folder):
                os.makedirs(folder)
        self._base = h5py.File(self._path, self._mode)

    def is_writable(self):
        return self._mode in ('a', 'w') and self.is_open

    def close(self):
        if self.is_open:
            self._base.close()
            self._base = None

    def _save_np(self, group, key, value, attrs=None):
        assert(isinstance(value, np.ndarray))
        dataset = group.create_dataset(key, data=value)
        if attrs is not None:
            for k, v in attrs.items():
                dataset.attrs[k] = v
        return dataset

    def _save_item(self, group, key, value):
        if isinstance(value, np.ndarray):
            return group.create_dataset(key, data=value)
        elif key == 'attrs':
            if not hasattr(value, 'items'):
                raise ValueError('attrs value must have `items` attr')
            for k, v in value.items():
                group.attrs[k] = v
        elif hasattr(value, 'items'):
            try:
                subgroup = group.create_group(key)
                for k, v in value.items():
                    self._save_item(subgroup, k, v)
                return subgroup
            except Exception:
                if key in subgroup:
                    del subgroup[key]
                raise
        else:
            raise TypeError(
                'value must be numpy array or have `items` attr, got %s'
                % str(value))

    def __setitem__(self, key, value):
        self._assert_writable('Cannot set item in unwritable dataset')
        self._save_item(self._base, key, value)

    def __delitem__(self, key):
        self._assert_writable('Cannot delete item in unwritable dataset')
        del self._base[key]


class Hdf5AutoSavingManager(auto_save.AutoSavingManager):
    def __init__(self, path, saving_message=None):
        self._path = path
        if saving_message is None:
            saving_message = 'Creating data for %s' % path
        self._saving_message = saving_message

    @property
    def saving_message(self):
        return self._saving_message

    @property
    def path(self):
        return self._path

    def get_saving_dataset(self, mode='a'):
        return Hdf5Dataset(self.path, mode)
