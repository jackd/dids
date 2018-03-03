

class InfiniteSet(object):

    def __len__(self):
        raise ValueError('length of infinite set is infinite')

    def __iter__(self):
        raise ValueError('cannot iterate over the infinite set')


class _EntireSet(InfiniteSet):
    def __contains__(self, key):
        return True

    def intersection(self, other):
        if isinstance(other, (set, frozenset, InfiniteSet)):
            return other
        else:
            return set(other)

    def union(self, other):
        return self

    def copy(self):
        return self

    def add(self):
        pass

    def issubset(self, other):
        return other is _EntireSet

    def issuperset(self, other):
        return True

    def difference(self, other):
        return negative_set(other)

    def symmetric_difference(self, other):
        return negative_set(other)

    def isdisjoint(self, other):
        return len(other) == 0


entire_set = _EntireSet()


class _NegativeSet(InfiniteSet):
    def __init__(self, negative_set):
        self._neg = negative_set

    def __contains__(self, key):
        return key not in self._neg

    def intersection(self, other):
        other = set(other)
        for key in self._neg:
            other.remove(key)
        return other

    def union(self, other):
        neg = set(self._neg)
        for k in other:
            if k in neg:
                neg.remove(k)
        return negative_set(neg)

    def copy(self):
        return _NegativeSet(self._neg.copy())

    def add(self, element):
        if element in self._neg:
            self._neg.remove(element)

    def remove(self, element):
        self._neg.add(element)

    def issubset(self, other):
        return other is _EntireSet or (
            other is _NegativeSet and other._neg.issubset(self._neg))

    def issuperset(self, other):
        return all(k not in other for k in self._neg)

    def difference(self, other):
        raise NotImplementedError()

    def symmetric_difference(self, other):
        return self.difference(other).union(other.difference(self))


def negative_set(neg):
    if neg is _EntireSet:
        return set()
    elif len(neg) == 0:
        return entire_set
    else:
        return _NegativeSet(neg)
