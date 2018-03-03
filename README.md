# Diction Interface to DataSets
Small `python` library for managing different sources in a dataset and interfacing with them as dictionaries.

Manages mapping, combining and saving/loading to file.

## Using Datasets
Datasets are designed to implement much of the standard python `dict` class's interface. Additionally, they are designed to be used in `with` context blocks.

```
d0 = Dataset.from_dict({'x': 1, 'y': 3})
d1 = Dataset.from_function(lambda x: x*3)

zipped = Dataset.zip(d0, d1)
with zipped:
    print(zipped['x'])           # (1, 'xxx')
    print(zipped['y'])           # (3, 'yyy')
    print('x' in zipped)         # True
    print('z' in zipped)         # False
    print(tuple(zipped.keys()))  # ('x', 'y'), or possibly ('y', 'x')
    try:
        print(zipped['z'])       # KeyError
    except KeyError:
        print('"z" not in zipped')
```

While not all datasets require use inside `with` blocks, it is highly recommended client code use them in such a way such that implementations can later be changed to require this. For example, `WrappedDictDataset`s do not require opening/closing. The source of the dataset may later be changed to a `JsonDataset`, which does. Code that runs without a `with` block will work for a `WrappedDictDataset`, but not a `JsonDataset`.

## Saving/loading
A number of implementations exist for writing/loading from file and are included in `file_io`. Currently these include:
* `json`
* `numpy`
* `hdf5`

## Implementing your own Dataset
Most datasets can be formed by a combination of mapping, key mapping and combining simpler datasets, or wrapping base dictionaries. If you do need to implement your own - e.g. for loading from a custom data format file, `UnwritableDataset` is the base class to extend if writing is not required. Extensions must implement only `__getitem__` and `keys` at the least.

If writing is required, `Dataset` can be extended. In addition to the method required for `UnwritableDataset`, `__setitem__` and `__delitem__` must be implemented.
