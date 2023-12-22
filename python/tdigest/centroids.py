import numpy as np
import numba as nb

from .centroid import Centroid
from . import centroid


MAX_SIZE = 100

Centroids = np.dtype([
    ("array", Centroid, MAX_SIZE),
    ("size", np.uint64),
])


@nb.njit(cache=True)
def new() -> np.ndarray[Centroids]:
    out = np.zeros(1, dtype=Centroids)
    self = out[0]
    self["size"] = 0
    return out


@nb.njit(cache=True)
def push_back(self: Centroids, value: Centroid) -> None:
    if self["size"] >= self["array"].size:
        return
    self["array"][self["size"]] = value
    self["size"] += 1


@nb.njit(cache=True)
def capacity(self: Centroids) -> np.uint64:
    return self["array"].size


@nb.njit(cache=True)
def size(self: Centroids) -> np.uint64:
    return self["size"]


@nb.njit(cache=True)
def sort(self: Centroids) -> None:
    array = self["array"][:self["size"]]
    for i in range(1, len(array)):
        j = i
        while j > 0 and not array[j - 1]["mean"] < array[j]["mean"]:
            array[j - 1], array[j] = array[j], array[j - 1]
            j -= 1


def repr_(self: Centroids) -> str:
    ss = [
        centroid.repr_(c)
        for c in self["array"][:self["size"]]
    ]
    return f"[{', '.join(ss)}]"
