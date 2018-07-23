#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
# import itertools
# from dids.file_io.hdf5 import Hdf5Dataset
# from dids.file_io.hdf5 import Hdf5ChildDataset
from dids.file_io.hdf5 import NestedHdf5Dataset

path = os.path.join(
    os.path.realpath(os.path.dirname(__file__)), 'example.hdf5')

if os.path.isfile(path):
    os.remove(path)


items = (
    (('a', 'a', 'a'), {
        'x': np.zeros((3, 2), dtype=np.float32),
        'y': np.zeros((4, 5), dtype=np.float32),
    }),
    (('a', 'a', 'b'), {
        'x': np.ones((3, 2), dtype=np.float32),
        'y': np.ones((4, 5), dtype=np.float32),
    })
)

ds = NestedHdf5Dataset(path=path, depth=3, mode='a')

with ds:
    ds.save_items(items, overwrite=True)

with ds:
    print(tuple(ds.keys()))
    for k, v in ds.items():
        print(k, v.keys())
