from dids.core import DelegatingDataset


def _nested_contains(key, group, depth):
    for k in key[:-1]:
        if k not in group:
            return False
        group = group[k]
    return key[-1] in group


def _nested_items(group, depth):
    if depth == 1:
        for key, value in group.items():
            yield (key,), value
    else:
        for key, value in group.items():
            for keys, value in _nested_items(value, depth-1):
                yield (key,) + keys, value


def _nested_keys(group, depth):
    if depth == 1:
        for key in group.keys():
            yield (key,)
    else:
        for key, value in group.items():
            for keys in _nested_keys(value, depth-1):
                yield (key,) + keys


def _nested_values(group, depth):
    if depth == 1:
        for value in group.values():
            yield value
    else:
        for value in group.values():
            for subval in _nested_values(value, depth-1):
                yield subval


def _nested_length(group, depth):
    if depth == 1:
        return len(group)
    else:
        return sum(_nested_length(v, depth-1) for v in group.values())


class NestedDataset(DelegatingDataset):
    def __init__(self, base, depth):
        if base is None:
            raise ValueError('base cannot be None')
        if depth is None or depth < 2:
            raise ValueError('base must be a positive integer')
        self._depth = depth
        super(NestedDataset, self).__init__(base)

    def keys(self):
        return _nested_keys(self._base, self._depth)

    def values(self):
        return _nested_values(self._base, self._depth)

    def items(self):
        return _nested_items(self._base, self._depth)

    def __len__(self):
        return _nested_length(self._base, self._depth)

    def __contains__(self, key):
        return _nested_contains(key, self._base, self._depth)

    def __getitem__(self, key):
        self._assert_valid_key(key)
        base = self._base
        for k in key:
            base = base[k]
        return base

    def __setitem__(self, key, value):
        self._assert_writable('Cannot set value of unwritable dataset')
        self._assert_valid_key(key)
        base = self._base
        for k in key[:-1]:
            base = base.setdefault(k, {})
        base[key[-1]] = value

    def __delitem__(self, key):
        self._assert_writable('Cannot delete item from unwritable dataset')
        self._assert_valid_key(key)
        base = self._base
        for k in key[:-1]:
            base = self._base[k]
        del base[key[-1]]

    def _assert_valid_key(self, key):
        if not (isinstance(key, tuple) and len(key) == self._depth):
            raise KeyError('key must be tuple of length %d, got "%s"'
                           % (self._depth, key))
