import errors
import sets
from progress.bar import IncrementalBar


class DummyBar(object):
    def next(self):
        pass

    def finish():
        pass


class LengthedGenerator(object):
    """Generator with an efficient, fixed length."""
    def __init__(self, gen, gen_len):
        self._gen = gen
        self._len = gen_len

    def __iter__(self):
        return iter(self._gen)

    def __len__(self):
        return self._len


class Dataset(object):
    """
    Abstract base class for dict-like interface with convenient wrapping fns.

    Concrete implementations must implement `keys` and `__getitem__`.

    Assumed to be used in a `with` clause, e.g.
    ```
    with MyDataset() as dataset:
        do_stuff_with(dataset)
    ```

    Default `__enter__`/`__exit__` implementations do nothing, but should be
    overriden if accessing files.
    """

    @property
    def is_open(self):
        """Flag indicating whether this dataset is open for reading."""
        return True

    def keys(self):
        self._assert_open('Cannot get keys from closed dataset')
        raise NotImplementedError('Abstract method')

    def __getitem__(self, key):
        self._assert_open('Cannot get item from closed dataset')
        raise NotImplementedError('Abstract method')

    @property
    def is_writable(self):
        """
        Flag indicating whether this dataset is currently writable.

        `__setitem__`, `__delitem__` will fail if not.
        """
        return False

    def __setitem__(self, key, value):
        self._assert_writable('Cannot save item to unwritable dataset')
        raise NotImplementedError('Abstract method')

    def __delitem__(self, key):
        self._assert_writable('Cannot delete item from unwritable dataset')
        raise NotImplementedError('Abstract method')

    def _assert_writable(self, message=None):
        if not self.is_writable:
            raise errors.UnwritableDatasetError(message)

    def _assert_open(self, message=None):
        if not self.is_open:
            raise errors.ClosedDatasetError(message)

    def __contains__(self, key):
        return key in self.keys()

    def __iter__(self):
        return iter(self.keys())

    def values(self):
        return (self[k] for k in self.keys())

    def items(self):
        return ((k, self[k]) for k in self.keys())

    def __len__(self):
        return len(self.keys())

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def open(self):
        pass

    def close(self):
        pass

    def get(self, key, default_value):
        try:
            return self[key]
        except errors.ClosedDatasetError:
            raise
        except KeyError:
            return default_value

    def pop(self, key, default_value):
        try:
            x = self[key]
            del self[key]
            return x
        except errors.UnwritableDatasetError:
            raise
        except errors.ClosedDatasetError:
            raise
        except KeyError:
            return default_value
        except IOError:
            return default_value

    def to_dict(self):
        return {k: v for k, v in self.items()}

    def subset(self, keys, check_present=True):
        return DataSubset(self, keys, check_present=check_present)

    def map(self, map_fn):
        return MappedDataset(self, map_fn)

    def map_keys(self, key_fn, inverse_fn=None):
        return KeyMappedDataset(self, key_fn, inverse_fn)

    def save_dataset(self, dataset, overwrite=False, show_progress=True,
                     message=None):
        if not self.is_open:
            raise IOError('Cannot save to non-open dataset.')
        keys = dataset.keys()
        if not overwrite:
            keys = [k for k in keys if k not in self]
        if len(keys) == 0:
            return
        if message is not None:
            print(message)
        bar = IncrementalBar(max=len(keys)) if show_progress else DummyBar()
        for key in keys:
            bar.next()
            if key in self:
                del self[key]
            value = dataset[key]
            self[key] = value
        bar.finish()

    def save_items(self, items, overwrite=False, show_progress=True):
        if not self.is_open:
            raise IOError('Cannot save to non-open dataset.')
        if show_progress:
            if hasattr(items, '__len__'):
                bar = IncrementalBar(max=len(items))
            else:
                bar = IncrementalBar()
        else:
            bar = DummyBar()
        for key, value in items:
            bar.next()
            if key in self:
                if overwrite:
                    del self[key]
                else:
                    continue
            self[key] = value
        bar.finish()

    @staticmethod
    def dict(**datasets):
        return DictDataset(**datasets)

    @staticmethod
    def zip(*datasets):
        return ZippedDataset(*datasets)

    @staticmethod
    def from_dict(dictionary):
        return WrappedDictDataset(dictionary)

    @staticmethod
    def wrapper(dictish):
        return DelegatingDataset(dictish)

    @staticmethod
    def from_function(key_fn):
        return FunctionDataset(key_fn)


class UnwritableDataset(Dataset):
    @property
    def is_writable(self):
        return False

    def __setitem__(self, key, value):
        raise errors.UnwritableDatasetError(
            'Cannot set item of unwritable dataset')

    def __delitem__(self, key, value):
        raise errors.UnwritableDatasetError(
            'Cannot delete item from unwritable dataset')

    def pop(self, key, default_value):
        raise errors.UnwritableDatasetError(
            'Cannot pop item from unwritable dataset')


class DelegatingDataset(Dataset):
    """
    Minimal wrapping implementation that wraps another dict-like object.

    Wrapped object must have a `keys` and `__getitem__` property. This class
    redirects to those, plus `open` and `close` methods if they exist.

    See also `WrappedDictDataset` for a similar implementation that wraps
    additional methods.
    """
    def __init__(self, base_dataset):
        if base_dataset is None:
            raise ValueError('`base_dataset` cannot be None')
        self._base = base_dataset

    @property
    def is_open(self):
        if hasattr(self._base, 'is_open'):
            return self._base.is_open
        else:
            return True

    @property
    def is_writable(self):
        if hasattr(self._base, 'is_writable'):
            return self._base.is_writable
        else:
            return False

    def keys(self):
        return self._base.keys()

    def __getitem__(self, key):
        return self._base[key]

    def open(self):
        if hasattr(self._base, 'open'):
            self._base.open()

    def close(self):
        if hasattr(self._base, 'close'):
            self._base.close()

    def __setitem__(self, key, value):
        self._base[key] = value

    def __delitem__(self, key):
        del self._base[key]


class WrappedDictDataset(DelegatingDataset):
    """Similar to DelegatingDataset, though redirects more methods."""

    def __contains__(self, key):
        return key in self._base

    def __iter__(self):
        return iter(self._base)

    @property
    def is_open(self):
        return True

    def values(self):
        return self._base.values()

    def items(self):
        return self._base.items()

    def is_writable(self):
        return True

    def __setitem__(self, key, value):
        self._base[key] = value

    def __delitem__(self, key):
        del self._base[key]


def key_intersection(keys_iterable):
    s = sets.entire_set
    for keys in keys_iterable:
        if isinstance(keys, sets.InfiniteSet):
            s = keys.intersection(s)
        else:
            s = s.intersection(keys)
    return s


class CompoundDataset(Dataset):
    """
    A dataset combining a number of datasets.

    The result of combining two datasets with the same keys is a dataset
    with the same keys and values equal to a dictionary of the dataset values.

    e.g.
    ```
    keys = ('hello', 'world')
    x = Dataset.from_function(lambda x: len(x), ('hello', 'world'))
    y = Dataset.from_function(lambda x: x*2, ('world',))
    xy = CompoundDataset(first=x, second=y)
    print(xy.keys())  # 'world'
    print(xy['world'])  # {'first': 5, 'second': 'worldworld'}
    ```
    """
    def __init__(self, **datasets):
        if not all(isinstance(d, Dataset) for d in datasets.values()):
            raise TypeError('All values of `dataset_dict` must be `Dataset`s')
        self._dataset_dict = datasets

    @property
    def is_open(self):
        return all(d.is_open for d in self.datasets)

    @property
    def datasets(self):
        raise NotImplementedError('Abstract method')

    def keys(self):
        try:
            return key_intersection(d.keys() for d in self.datasets)
        except errors.ClosedDatasetError:
            raise errors.ClosedDatasetError(
                'Cannot get keys for closed dataset')

    def __contains__(self, key):
        try:
            return all(key in d for d in self.datasets)
        except errors.ClosedDatasetError:
            raise errors.ClosedDatasetError(
                'Cannot check membership for closed dataset')

    def open(self):
        for v in self.datasets:
            v.open()

    def close(self):
        for v in self.datasets:
            v.close()

    def __delitem__(self, key):
        for dataset in self.datasets:
            del dataset[key]


class DictDataset(CompoundDataset):
    def __init__(self, **datasets):
        self._datasets = datasets

    def __getitem__(self, key):
        return {k: v[key] for k, v in self._datasets.items()}

    def __setitem__(self, key, value):
        if not hasattr(value, 'items'):
            raise TypeError('value must have items for CompoundDataset')
        for value_key, dataset in self._datasets.items():
            dataset[key] = value[value_key]

    @property
    def datasets(self):
        return tuple(self._datasets.values())


class ZippedDataset(CompoundDataset):
    def __init__(self, *datasets):
        self._datasets = datasets

    @property
    def datasets(self):
        return self._datasets

    def __getitem__(self, key):
        return tuple(d[key] for d in self._datasets)

    def __setitem__(self, key, value):
        if not hasattr(value, '__iter__'):
            raise TypeError('value must be iterable for ZippedDataset')
        for dataset, v in zip(self._datasets, value):
            dataset[key] = v


class MappedDataset(DelegatingDataset):
    """Dataset representing a mapping applied to a base dataset."""
    def __init__(self, base_dataset, map_fn):
        super(MappedDataset, self).__init__(base_dataset)
        self._map_fn = map_fn

    def __contains__(self, key):
        return key in self._base

    def __getitem__(self, key):
        return self._map_fn(self._base[key])

    def __len__(self):
        return len(self._base)

    def subset(self, keys, check_present=True):
        return self._base.subset(keys, check_present).map(self._map_fn)

    @property
    def is_writable(self):
        return False


class DataSubset(DelegatingDataset):
    """Dataset with keys constrained to a given subset."""
    def __init__(self, base_dataset, keys, check_present=True):
        if base_dataset is None:
            raise ValueError('`base_dataset` cannot be None')
        self._check_present = check_present
        self._keys = frozenset(keys)
        super(DataSubset, self).__init__(base_dataset)
        if self.is_open:
            self._check_keys()

    def _check_keys(self):
        if self._check_present:
            for key in self._keys:
                if key not in self._base:
                    raise KeyError('key %s not present in base' % key)

    def subset(self, keys, check_present=True):
        if check_present:
            for key in keys:
                if key not in self._keys:
                    raise KeyError('key %s not present in base' % key)

        return DataSubset(self._base, keys, check_present and not self.is_open)

    def keys(self):
        self._assert_open('Cannot get keys for closed dataset')
        return self._keys

    def __getitem__(self, key):
        self._assert_open('Cannot get item from closed dataset')
        if key not in self._keys:
            raise errors.invalid_key_error(self, key)
        return self._base[key]

    def __setitem__(self, key, value):
        self._assert_writable('Cannot set item for unwritable dataset')
        if key in self._keys:
            self._base[key] = value
        else:
            raise errors.invalid_key_error(self, key)

    def __delitem__(self, key):
        self._assert_writable('Cannot delete item from unwritable dataset')
        if key in self._keys:
            del self._base[key]
        else:
            raise errors.invalid_key_error(self, key)

    def open(self):
        super(DataSubset, self).open()
        self._check_keys()


class FunctionDataset(UnwritableDataset):
    """Dataset which wraps a function."""
    def __init__(self, key_fn):
        self._key_fn = key_fn

    def keys(self):
        return sets.entire_set

    def __contains__(self, key):
        return True

    def __getitem__(self, key):
        return self._key_fn(key)

    @property
    def is_open(self):
        return True


class KeyMappedDataset(Dataset):
    """
    Dataset with keys mapped.

    e.g.
    ```
    base = Dataset.from_function(lambda x: len(x), ('hello', 'world'))
    key_mapped = KeyMappedDataset(base, lambda x: x[5:])
    print(key_mapped['ahoy hello'])  # 5
    ```
    """
    def __init__(self, base_dataset, key_fn, inverse_fn=None):
        self._base = base_dataset
        self._key_fn = key_fn
        self._inverse_fn = inverse_fn

    def keys(self):
        if self._inverse_fn is None:
            raise errors.unknown_keys_error(self)
        else:
            return (self._inverse_fn(k) for k in self._base.keys())

    def __len__(self):
        if self._keys is None:
            return len(self._base)
        else:
            return len(self._keys)

    def __getitem__(self, key):
        mapped_key = self._key_fn(key)
        try:
            return self._base[mapped_key]
        except KeyError:
            raise KeyError('%s -> %s not in base dataset ' % (key, mapped_key))

    def __contains__(self, key):
        return self._key_fn(key) in self._base

    def open(self):
        self._base.open()

    def close(self):
        self._base.close()

    def __delitem__(self, key):
        del self._base[self._key_fn(key)]

    @property
    def is_open(self):
        return self._base.is_open

    @property
    def is_writable(self):
        return self._base.is_writable

    def __setitem__(self, key, value):
        self._base[self._key_fn(key)] = value


class PrioritizedDataset(UnwritableDataset):
    def __init__(self, *datasets):
        self._datasets = datasets

    def __getitem__(self, key):
        for d in self._datasets:
            if key in d:
                return d[key]
        raise errors.invalid_key_error(self, key)

    def __contains__(self, key):
        return any(key in d for d in self._datasets)

    def keys(self):
        ds = self._datasets
        return ds[0].keys().union(d.keys() for d in ds)

    @property
    def is_open(self):
        return all(d.is_open for d in self._datasets)

    def open(self):
        for d in self._datasets:
            d.open()

    def close(self):
        for d in self._datasets:
            d.close()


class BiKeyDataset(Dataset):
    """
    Class for grouping multiple datasets into one with an additional key.

    e.g.
    ```
    cars_ds = DictDataset({'c1': 'car1.png', 'c2': 'car2.png'}
    tables_ds = DictDataset({'t1': 't1.png', 't2': 't2.png'})

    obj_ds = BiKeyDataset({'car': cars_ds, 'table': tables_ds})
    with obj_ds:
        print(obj_ds[('car', 'c1')])  # 'car1.png'
    ```
    """
    def __init__(self, **datasets):
        self._datasets = datasets

    def __getitem__(self, key):
        k0, k1 = key
        try:
            dataset = self._datasets[k0]
        except KeyError:
            raise KeyError('No dataset at first key %s' % k0)
        try:
            return dataset[k1]
        except KeyError:
            raise KeyError('No entry in sub-dataset, %s' % str(key))

    def __contains__(self, key):
        k0, k1 = key
        return k0 in self._datasets and k1 in self._datasets[k0]

    def keys(self):
        for k0, dataset in self._datasets.items():
            for k1 in dataset:
                yield (k0, k1)
        # return itertools.chain(*(
        #     (k0, k1) for k1 in dataset)
        #     for k0, dataset in self._datasets.items())

    def open(self):
        for dataset in self._datasets.values():
            dataset.open()

    def close(self):
        for dataset in self._datasets.values():
            dataset.close()

    @property
    def is_open(self):
        return all(d.is_open for d in self._datasets.values())

    def values(self):
        for dataset in self._datasets.values():
            for v in dataset.values():
                yield v
        # return itertools.chain(
        #     *(d.values() for d in self._datasets.values()))

    def __len__(self):
        return sum(len(d) for d in self._datasets.values())

    def items(self):
        for k0, dataset in self._datasets.items():
            for k1, v in dataset.items():
                yield (k0, k1), v

    @property
    def is_writable(self):
        return all(d.is_writable for d in self._datasets.values())

    def __setitem__(self, key, value):
        k0, k1 = key
        self._datasets[k0][k1] = value

    def __delitem__(self, key):
        k0, k1 = key
        del self._datasets[k0][k1]
