import tarfile
import dids.core as core


class TarFileDataset(core.Dataset):
    def __init__(self, path, mode='r'):
        self._file = None
        self._path = path
        self._mode = mode

    @property
    def path(self):
        return self._path

    def _open_resource(self):
        if self._file is None:
            self._file = tarfile.TarFile(self._path, self._mode)
        self._keys = None

    def _close_resource(self):
        if self._file is None:
            return
        self._file.close()
        self._file = None

    def keys(self):
        self._assert_open('Cannot get keys of closed dataset')
        if self._keys is None:
            self._keys = frozenset(self._file.getnames())
        return self._keys

    def __getitem__(self, key):
        return self._file.extractfile(self._file.getmember(key))

    def __setitem__(self, key, value):
        self._assert_writable('Cannot __setitem__ if dataset is not writable')
        self._file.writestr(key, value.read())

    @property
    def is_open(self):
        return self._file is not None

    @property
    def is_writable(self):
        return self._mode in ('a', 'w')
