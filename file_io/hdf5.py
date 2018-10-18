import os
import numpy as np
import h5py
import dids.core as core
import dids.auto_save as auto_save
from dids.nest import NestedDataset


def _save_item(group, key, value, compression=None):
    if isinstance(value, np.ndarray):
        return group.create_dataset(key, data=value, compression=compression)
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
                _save_item(subgroup, k, v)
            return subgroup
        except Exception:
            if subgroup is not None and key in subgroup:
                del subgroup[key]
            raise
    else:
        raise TypeError(
            'value must be numpy array or have `items` attr, got %s'
            % str(value))


class Hdf5Resource(core.Resource):
    def __init__(self, path, mode='r'):
        self._path = path
        self._mode = mode

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

    @property
    def is_open(self):
        return self._base is not None


class Hdf5ChildResource(core.Resource):
    def __init__(self, parent, subpath):
        self._parent = parent
        self._is_open = False

    def _open_resource(self):
        if self.is_open():
            raise RuntimeError('Cannot open resource: already open.')
        self._parent.open_connection(self)
        self._is_open = True

    def _close_resourece(self):
        if not self.is_open():
            raise RuntimeError('Cannot close resource: not open.')
        self._parent.close_resource(self)
        self._is_open = False

    def is_open(self):
        return self._is_open


class Hdf5Dataset(Hdf5Resource, core.WrappedDictDataset):
    def __init__(self, path, mode='r', compression=None):
        self._base = None
        self._compression = compression
        Hdf5Resource.__init__(self, path, mode)

    @property
    def compression(self):
        return self._compression

    def __setitem__(self, key, value):
        self._assert_writable('Cannot set item in unwritable dataset')
        _save_item(self._base, key, value, compression=self.compression)

    def __delitem__(self, key):
        self._assert_writable('Cannot delete item in unwritable dataset')
        del self._base[key]

    @property
    def path(self):
        return self._path

    def is_writable(self):
        return self._mode in ('a', 'w') and self.is_open


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


# def _nested_hdf5_child_dataset(parent, subpath, depth):
#     print('--')
#     print(parent, subpath)
#     base = Hdf5ChildDataset(parent, subpath)
#     if depth == 1:
#         return base
#     else:
#         return BiKeyDataset(base).map_keys(
#             lambda x: (x[0], x[1:]), lambda x: (x[0],) + x[1]).map(
#             lambda x: _nested_hdf5_child_dataset(x._base, depth-1))
#
#
# def nested_hdf5_dataset(depth, path, mode='r'):
#     return NestedDataset(Hdf5Dataset(path, mode), depth)


# def nested_hdf5_group_dataset(depth, group):
#     base = Hdf5GroupDataset(group)
#     if depth == 1:
#         return base
#
#     def child_fn(child, depth):
#         base = Hdf5GroupDataset(child)
#         if depth == 1:
#             base = Hdf5GroupDataset(base)
#         else:
#             return nested_hdf5_group_dataset(depth, group)
#
#     return NestedDataset(base, depth, child_fn)
#
#
# def nested_hdf5_dataset(depth, path, mode='r'):
#     base = Hdf5Dataset(path, mode)
#     if depth == 1:
#         return base

    # def child_fn(child, depth):
    #     base = Hdf5GroupDataset(child)
    #     if depth == 1:
    #         return base
    #     else:
    #         return nested_hdf5_group_dataset(base, depth)

    # return NestedDataset(base, depth, nested_hdf5_group_dataset)


# NestedHdf5Dataset = nested_hdf5_dataset

class NestedHdf5Dataset(NestedDataset):
    def __init__(self, depth, path, mode='r', compression=None):
        base = Hdf5Dataset(path, mode=mode, compression=compression)
        super(NestedHdf5Dataset, self).__init__(base, depth)

    def __setitem__(self, key, value):
        self._assert_writable('Cannot set value of unwritable dataset')
        self._assert_valid_key(key)
        base = self._base._base
        for k in key[:-1]:
            base = base.require_group(k)
        _save_item(base, key[-1], value, compression=self.compression)

    def get_child(self, k0):
        return Hdf5ChildDataset(self, k0)

    def children_keys(self):
        return self._base.keys()


# class NestedHdf5Dataset(Hdf5Dataset):
#     def __init__(self, depth, path, mode='r'):
#         self._depth = depth
#         super(NestedHdf5Dataset, self).__init__(path, mode=mode)
#
#
nested_hdf5_dataset = NestedHdf5Dataset


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

    def __contains__(self, key):
        return 0 <= key < len(self._base)

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
            return nested_hdf5_dataset(self._nested_depth, self.path, mode)
        else:
            return Hdf5Dataset(self.path, mode)
