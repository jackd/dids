import os
import shutil
import uuid


class TempDir(object):
    def __init__(self, path):
        self._path = path

    def __enter__(self):
        if os.path.isdir(self._path):
            raise IOError('Directory already exists at %s' % self._path)
        os.makedirs(self._path)
        return self._path

    def __exit__(self, *args, **kwargs):
        shutil.rmtree(self._path)


class TempPath(object):
    def __init__(self, folder='/tmp', extension=''):
        self.folder = folder
        self.extension = extension
        self.path = None

    def __enter__(self):
        self.open()
        return self.path

    def __exit__(self, *args, **kwargs):
        self.close()

    def open(self):
        self.path = get_random_path(self.folder, self.extension)
        while os.path.isfile(self.path):
            self.path = get_random_path(self.folder, self.extension)

    def close(self):
        if self.path is None:
            raise RuntimeError('Cannot close unopened TempPath')
        os.remove(self.path)


def get_random_path(folder, extension):
    return os.path.join(folder, '%s%s' % (uuid.uuid4().hex, extension))
