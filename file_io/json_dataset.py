import os
import json
import dids.core as core
import dids.auto_save as auto_save


class JsonDataset(core.WrappedDictDataset):
    def __init__(self, path, mode='r'):
        self._path = path
        self._mode = mode
        self._base = None

    @property
    def is_open(self):
        return self._base is not None

    def _open_resource(self):
        if self._mode == 'r' and not os.path.isfile(self._path):
            raise IOError(
                'Cannot load json data: file does not exist at %s self._path')
        if self._mode in ('r', 'a') and os.path.isfile(self._path):
            with open(self._path, 'r') as fp:
                try:
                    self._base = json.load(fp)
                except ValueError:
                    raise ValueError('Invalid json data at %s' % self._path)
        else:
            self._base = {}

    @property
    def is_writable(self):
        return self._mode in ('a', 'w') and self.is_open

    def _close_resource(self):
        if self.is_writable:
            folder = os.path.dirname(self._path)
            if not os.path.isdir(folder):
                os.makedirs(folder)
            with open(self._path, 'w') as fp:
                json.dump(self._base, fp)
        self._base = None

    def __setitem__(self, key, value):
        self._assert_writable('Cannot set value of unwritable dataset')
        self._base[key] = value

    def __delitem__(self, key):
        self._assert_writable('Cannot delete item from unwritable dataset')
        del self._base[key]


class JsonAutoSavingManager(auto_save.AutoSavingManager):
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
        return JsonDataset(self.path, mode)
