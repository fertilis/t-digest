import numpy as np
import numba as nb

Centroid = np.dtype([
    ("mean", np.float64),
    ("weight", np.float64),
])


@nb.njit(cache=True)
def new(mean: np.float64, weight: np.float64) -> np.ndarray[Centroid]:
    out = np.zeros(1, dtype=Centroid)
    self = out[0]
    self["mean"] = mean
    self["weight"] = weight
    return out


@nb.njit(cache=True)
def add(self: Centroid, sum_: np.float64, weight: np.float64):
    sum_ += self["mean"] * self["weight"]
    self["weight"] += weight
    self["mean"] = sum_ / self["weight"]
    return sum_


@nb.njit(cache=True)
def is_less(self: Centroid, other: Centroid) -> np.bool_:
    return np.bool_(self["mean"] < other["mean"])


def repr_(self: Centroid) -> str:
    return f"({float(self['mean']):.2f}, {float(self['weight']):.2f})"
