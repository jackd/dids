import os
from plyfile import PlyData
from .file_dataset import FileDataset


class PlyDataset(FileDataset):
    def __init__(self, root_dir, mode='r'):
        if mode != 'r':
            raise NotImplementedError('only mode r implemented')
        super(PlyDataset, self).__init__(root_dir, mode)

    def __getitem__(self, key):
        return PlyData.read(self.path(key))

    def __contains__(self, key):
        return os.path.isfile(self.path(key))

    def __setitem__(self, key, value):
        self._assert_writable('Cannot set item of unwritable dataset')
        raise NotImplementedError('ply file writing not implemented.')

    def __delitem__(self, key):
        self._assert_writable('Cannot delete item from unwritable dataset')
        raise NotImplementedError('ply file deletion not implemented.')
        # os.remove(self.path(key))
