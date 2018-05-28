import os
import numpy as np
import h5py
import dids.core as core
import dids.auto_save as auto_save


def _nested_items(group, depth):
    if depth == 1:
        for key, value in group.items():
            yield (key,), value
    else:
        for key, value in group.items():
            for keys, value in _nested_items(value, depth-1):
                yield (key,) + keys, value


def _nested_keys(group, depth):
    if depth == 1:
        for key in group.keys():
            yield (key,)
    else:
        for key, value in group.items():
            for keys in _nested_keys(value, depth-1):
                yield (key,) + keys


def _nested_values(group, depth):
    if depth == 1:
        for value in group.values():
            yield value
    else:
        for value in group.values():
            for subval in _nested_values(value, depth-1):
                yield subval


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

    def is_writable(self):
        return self._mode in ('a', 'w') and self.is_open

    def _open_resource(self):
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

    def _close_resource(self):
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
            subgroup = None
            try:
                subgroup = group.create_group(key)
                for k, v in value.items():
                    self._save_item(subgroup, k, v)
                return subgroup
            except Exception:
                if subgroup is not None and key in subgroup:
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


class NestedHdf5Dataset(Hdf5Dataset):
    def __init__(self, path, mode='r', depth=2):
        super(NestedHdf5Dataset, self).__init__(path, mode)
        self._depth = depth

    def __iter__(self):
        return iter(self.keys())

    def items(self):
        return _nested_items(self._base, self._depth)

    def values(self):
        return _nested_values(self._base, self._depth)

    def keys(self):
        return _nested_keys(self._base, self._depth)

    def _assert_valid_key(self, key):
        if not (isinstance(key, tuple) and len(key) == self._depth):
            raise KeyError('key must be tuple of length %d, got "%s"'
                           % (self._depth, key))

    def __getitem__(self, key):
        self._assert_valid_key(key)
        return self._base[os.path.join(*key)]

    def __contains__(self, key):
        self._assert_valid_key(key)
        return os.path.join(*key) in self._base

    def __setitem__(self, key, value):
        self._assert_writable('Cannot set item in unwritable dataset')
        self._assert_valid_key(key)
        self._save_item(self._base, os.path.join(*key), value)

    def __delitem__(self, key):
        self._assert_writable('Cannot delete item in unwritable dataset')
        self._assert_valid_key(key)
        del self._base[os.path.join(*key)]


class Hdf5ChildDataset(Hdf5Dataset):
    def __init__(self, parent, subpath):
        self._parent = parent
        self._subpath = subpath
        self._base = None

    def _create_self(self):
        self._parent._base.create_group(self._subpath)

    @property
    def path(self):
        return self._parent.path

    @property
    def subpath(self):
        return self._subpath

    def is_writable(self):
        return self._parent.is_writable()

    def _open_resource(self):
        if self.is_open:
            raise IOError('Hdf5Dataset already open')
        self._parent.open_connection(self)
        if self._subpath not in self._parent:
            self._create_self()
        self._base = self._parent[self._subpath]

    def _close_resource(self):
        if self.is_open:
            self._base = None


class Hdf5ArrayDataset(Hdf5ChildDataset):
    def __init__(self, parent, subpath, shape=None, dtype=None):
        self._shape = shape
        self._dtype = dtype
        super(Hdf5ArrayDataset, self).__init__(parent, subpath)

    def _create_self(self):
        self._parent._base.create_dataset(
            self._subpath, shape=self._shape, dtype=self._dtype)

    def keys(self):
        return range(len(self._base))

    def __delitem__(self, key):
        raise NotImplementedError('Cannot delete item from array dataset')

    def __setitem__(self, key, value):
        self._assert_writable('Cannot set item in unwritable dataset')
        self._base[key] = value


class Hdf5AutoSavingManager(auto_save.AutoSavingManager):
    def __init__(self, path, saving_message=None, nested_depth=None):
        self._path = path
        if saving_message is None:
            saving_message = 'Creating data for %s' % path
        self._saving_message = saving_message
        self._nested_depth = nested_depth

    @property
    def saving_message(self):
        return self._saving_message

    @property
    def path(self):
        return self._path

    def get_saving_dataset(self, mode='a'):
        if hasattr(self, '_nested_depth') and self._nested_depth:
            return NestedHdf5Dataset(self.path, mode, depth=self._nested_depth)
        else:
            return Hdf5Dataset(self.path, mode)
