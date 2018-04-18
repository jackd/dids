import dids.core as core
import dids.errors as errors


class AutoSavingDataset(core.Dataset):
    def __init__(self, src, dst):
        if not all(hasattr(src, k) for k in ('items', '__getitem__')):
            raise TypeError('`src` must have `items` and `__getitem__` attrs')
        self._src = src
        self._dst = dst

    @property
    def src(self):
        """The source dataset this dataset gets data from."""
        return self._src

    @property
    def dst(self):
        """The destination dataset this dataset saves to."""
        return self._dst

    def unsaved_keys(self):
        return (k for k in self.src.keys() if k not in self.dst)

    def keys(self):
        return self.src.keys()

    def __contains__(self, key):
        return key in self.src

    def __getitem__(self, key):
        if key not in self.src:
            raise errors.invalid_key_error(self, key)
        if key in self.dst:
            return self.dst[key]
        else:
            value = self.src[key]
            self.dst.save_item(key, value)
            return value

    def save_all(self, overwrite=False, show_progress=True, message=None):
        self.dst.save_dataset(
            self.src, overwrite=overwrite, show_progress=show_progress,
            message=message)

    def _open_resource(self):
        self.src.open_connection(self)
        self.dst.open_connection(self)

    def _close_resource(self):
        self.dst.close_connection(self)
        self.src.close_connection(self)

    def subset(self, keys, check_present=True):
        src = self.src.subset(keys, check_present)
        dst = self.dst
        return AutoSavingDataset(src, dst)


class AutoSavingManager(object):
    def get_lazy_dataset(self):
        raise NotImplementedError('Abstract method')

    def get_saving_dataset(self, mode='r'):
        raise NotImplementedError('Abstract method')

    def get_auto_saving_dataset(self, mode='a'):
        return AutoSavingDataset(
            self.get_lazy_dataset(),
            self.get_saving_dataset(mode))

    @property
    def saving_message(self):
        return None

    def save_all(self, overwrite=False):
        with self.get_auto_saving_dataset('a') as ds:
            ds.save_all(overwrite=overwrite, message=self.saving_message)

    def get_saved_dataset(self):
        self.save_all()
        return self.get_saving_dataset(mode='r')
