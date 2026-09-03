"""``Bunch``: a dict whose keys are also attributes (the return type of the dataset loaders)."""

from __future__ import annotations

__all__ = ["Bunch"]


class Bunch(dict):
    """Container object exposing keys as attributes.

    ``Bunch`` extends :class:`dict` so that values can be accessed either with
    ``b["key"]`` or ``b.key``. Setting an attribute stores a key.

    Examples
    --------
    >>> from skroute.utils import Bunch
    >>> b = Bunch(a=1, b="two")
    >>> b.a, b["b"]
    (1, 'two')
    >>> b.c = [3]
    >>> sorted(b)
    ['a', 'b', 'c']
    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __dir__(self):
        return sorted(set(dir(type(self))) | set(self))

    def __repr__(self):
        return f"Bunch({', '.join(sorted(self))})"
