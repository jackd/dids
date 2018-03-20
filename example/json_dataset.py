import os
from dids.file_io.json_dataset import JsonDataset

path = os.path.join(
    os.path.realpath(os.path.dirname(__file__)), 'data.json')

ds = JsonDataset(path, mode='a')
with ds:
    if len(ds) == 0:
        ds['a'] = 'hello'
        ds['b'] = 'world'
    elif 'c' in ds:
        del ds['c']

ds = JsonDataset(path, mode='r')

print('With mode=\'r\'')
with ds:
    print('length: %d' % len(ds))
    print('values: %s' % str(tuple(ds.items())))
    try:
        ds['c'] = 'another'
        print('Wrote to dataset')
    except IOError:
        print('Attempted to write to dataset, but not writable')

ds = JsonDataset(path, 'a')
print('With mode=\'a\'')
with ds:
    try:
        ds['c'] = 'another'
        print('Wrote to dataset')
    except IOError:
        print('Attempted to write to dataset, but not writable')
