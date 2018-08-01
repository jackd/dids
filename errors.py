class UnknownKeysError(NotImplementedError):
    def __init__(self, dataset):
        super(UnknownKeysError, self).__init__(
            'Keys unknown. '
            'Specify with `dataset = dataset.subset(keys)` if known'
        )


class ModeError(ValueError):
    def __init__(self, mode):
        super(ModeError, self).__init__(
            '`mode` must be in ("r", "w", "a"), got %s' % mode)


class ClosedDatasetError(IOError):
    def __init__(self, message):
        super(ClosedDatasetError, self).__init__(message)


class UnwritableDatasetError(IOError):
    def __init__(self, message):
        super(UnwritableDatasetError, self).__init__(message)


def unknown_keys_error(dataset):
    return UnknownKeysError(dataset)


def mode_error(mode):
    return ModeError(mode)


def invalid_key_error(dataset, key):
    return KeyError('key %s not a valid key' % str(key))
