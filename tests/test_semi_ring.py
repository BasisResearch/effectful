import collections.abc

from effectful.ops.core import Operation, Term, evaluate, gensym
from effectful.ops.handler import fwd, handler


# https://stackoverflow.com/questions/2703599/what-would-a-frozen-dict-be
class SemiRingDict(collections.abc.Mapping):
    def __init__(self, *args, **kwargs):
        self._d = dict(*args, **kwargs)
        self._hash = None

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __getitem__(self, key):
        return self._d[key]

    def __hash__(self):
        # It would have been simpler and maybe more obvious to
        # use hash(tuple(sorted(self._d.iteritems()))) from this discussion
        # so far, but this solution is O(n). I don't know what kind of
        # n we are going to run into, but sometimes it's hard to resist the
        # urge to optimize when it will gain improved algorithmic performance.
        if self._hash is None:
            hash_ = 0
            for pair in self.items():
                hash_ ^= hash(pair)
            self._hash = hash_
        return self._hash

    def __add__(self, other):
        new_dict = self._d.copy()
        for key, value in other.items():
            if key in new_dict:
                new_dict[key] += value
            else:
                new_dict[key] = value
        return SemiRingDict(**new_dict)


@Operation
def Sum(e1, k, v, e2):
    raise NotImplementedError


@Operation
def Let(x, e1, e2):
    raise NotImplementedError


@Operation
def Record(**kwargs):
    raise NotImplementedError


@Operation
def Dict(*contents):
    raise NotImplementedError


@Operation
def Field(record, key):
    raise NotImplementedError


@Operation
def App(f, x):
    raise NotImplementedError


def is_value(v):
    return not (isinstance(v, Term) or isinstance(v, Operation))


def eager_dict(*contents):
    if all(is_value(v) for v in contents):
        if len(contents) % 2 != 0:
            raise ValueError("Dict requires an even number of arguments")

        kv = []
        for i in range(0, len(contents), 2):
            kv.append((contents[i], contents[i + 1]))
        return SemiRingDict(kv)
    else:
        return fwd(None)


def eager_record(**kwargs):
    if all(is_value(v) for v in kwargs.values()):
        return dict(**kwargs)
    else:
        return fwd(None)


def eager_app(f, x):
    if is_value(f) and is_value(x):
        return f(x)
    else:
        return fwd(None)


def eager_field(r, k):
    match r, k:
        case dict(), str():
            return r[k]
        case SemiRingDict(), _ if is_value(k):
            return r[k]
        case _:
            return fwd(None)


def eager_sum(e1, k, v, e2):
    match e1, e2:
        case SemiRingDict(), Term():
            new_d = SemiRingDict()
            for key, value in e1.items():
                new_d += handler({k: lambda: key, v: lambda: value})(evaluate)(e2)
            return new_d
        case SemiRingDict(), SemiRingDict():
            new_d = SemiRingDict()
            for _ in e1.items():
                new_d += e2
            return new_d
        case _:
            return fwd(None)


def eager_let(e1, x, e2):
    match e1, e2:
        case SemiRingDict(), Term():
            return handler({x: lambda: e1})(evaluate)(e2)
        case _, SemiRingDict():
            return e2
        case _:
            return fwd(None)


free = {
    Sum: lambda *args: Term(Sum, args, ()),
    Let: lambda x, e1, e2: Term(Let, (x, e1, e2), ()),
    Record: lambda **kwargs: Term(Record, (), list(kwargs.items())),
    Dict: lambda *contents: Term(Dict, contents, ()),
    Field: lambda r, k: Term(Field, (r, k), ()),
}

eager = {
    Dict: eager_dict,
    Record: eager_record,
    Sum: eager_sum,
    Field: eager_field,
    Let: eager_let,
}


def test_simple_sum():
    x = gensym(object)
    k = gensym(object)
    v = gensym(object)

    with handler(free), handler(eager):
        e = Sum(Dict("a", 1, "b", 2), k, v, Dict("v", v()))
        assert e["v"] == 3

    with handler(free), handler(eager):
        e = Let(Dict("a", 1, "b", 2), x, Field(x(), "b"))
        assert e == 2
