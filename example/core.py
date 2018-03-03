from dids.core import Dataset
# function example
base = Dataset.from_function(lambda x: 3*x)

with base:
    for k in [3, 2.7, 'hello world']:
        try:
            print('%s -> %s' % (k, base[k]))
        except ValueError:
            print('Error getting key %s' % k)
print('is "hello world" in base? %s' % ('hello world' in base))

subset = base.subset([3, 4, 7])

print('Subsets')
with subset:
    for k in [3, 4, 5]:
        if k in subset:
            print('%s -> %s' % (k, subset[k]))
        else:
            print('Key %s not in subset' % k)

print('Mapping')
base_mapped = base.map(lambda x: x**2)

k = 4
with base:
    print('%s -> %s' % (k, base_mapped[k]))

print('Zipping')
base_zipped = Dataset.zip(base, base_mapped)
with base_zipped:
    print('%s -> %s' % (k, base_zipped[k]))


print('Dict')
base_dict = Dataset.dict(x=base, y=base_mapped)
with base_dict:
    print('%s -> %s' % (k, base_dict[k]))


print('Keys display')
d0 = Dataset.from_dict({'x': 1, 'y': 3})
d1 = Dataset.from_function(lambda x: x*3)

zipped = Dataset.zip(d0, d1)
with zipped:
    print(zipped['x'])             # (1, 'xxx')
    print(zipped['y'])             # (3, 'yyy')
    print('x' in zipped)           # True
    print('z' in zipped)           # False
    print(tuple(zipped.keys()))    # ('x', 'y'), or possibly ('y', 'x')
    try:
        print(zipped['z'])         # KeyError
    except KeyError:
        print('"z" not in zipped')
