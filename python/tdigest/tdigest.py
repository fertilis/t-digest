import numpy as np
import numba as nb

from .centroids import Centroids
from .centroid import Centroid
from . import centroid
from . import centroids


TDigest = np.dtype([
    ("centroids", centroids.Centroids),
    ("max_size", np.uint64),
    ("sum", np.float64),
    ("count", np.float64),
    ("max", np.float64),
    ("min", np.float64),
])


@nb.njit(cache=True)
def new() -> np.ndarray[TDigest]:
    out = np.zeros(1, dtype=TDigest)
    self = out[0]
    self["centroids"] = centroids.new()[0]
    self["max_size"] = centroids.capacity(self["centroids"])
    self["sum"] = np.float64(0.0)
    self["count"] = np.float64(0.0)
    self["max"] = np.nan
    self["min"] = np.nan
    return out


@nb.njit(cache=True)
def add_value(self: np.ndarray[TDigest], value: np.float64):
    add_sorted_values(self, np.array([value], dtype=np.float64))


@nb.njit(cache=True)
def add_sorted_values(self: np.ndarray[TDigest], values: np.ndarray[np.float64]) -> None:
    if values.size == 0:
        return
    result: np.ndarray[TDigest] = new()
    result[0]["count"] = self[0]["count"] + values.size
    maybe_min = values[0]
    maybe_max = values[-1]
    if self[0]["count"] > 0:
        # We know that min_ and max_ are numbers
        result[0]["min"] = np.minimum(self[0]["min"], maybe_min)
        result[0]["max"] = np.maximum(self[0]["max"], maybe_max)
    else:
        # We know that min_ and max_ are NaN.
        result[0]["min"] = maybe_min
        result[0]["max"] = maybe_max

    compressed_a: np.ndarray[Centroids] = centroids.new()
    compressed: Centroids = compressed_a[0]

    k_limit = np.float64(1.0)
    q_limit_times_count: np.float64 = k_to_q(k_limit, self[0]["max_size"]) * result[0]["count"]
    k_limit += 1

    centroid_index = np.uint64(0)
    values_index = np.uint64(0)

    cur: np.ndarray[Centroid] = np.zeros(1, dtype=Centroid)
    if (
        centroid_index < self[0]["centroids"]["size"]
        and self[0]["centroids"]["array"][centroid_index]["mean"] < values[values_index]
    ):
        cur = self[0]["centroids"]["array"][centroid_index:centroid_index+np.uint64(1)]
        centroid_index += np.uint64(1)
    else:
        cur = centroid.new(values[values_index], 1.0)
        values_index += np.uint64(1)

    weight_so_far = cur[0]["weight"]

    # Keep track of sums along the way to reduce expensive floating points
    sums_to_merge = np.float64(0.0)
    weights_to_merge = np.float64(0.0)

    next_: np.ndarray[Centroid] = np.zeros(1, dtype=Centroid)
    while (
        centroid_index < self[0]["centroids"]["size"]
        or values_index < values.size
    ):
        if (
            centroid_index < self[0]["centroids"]["size"]
            and (
                values_index >= values.size
                or self[0]["centroids"]["array"][centroid_index]["mean"] < values[values_index]
            )
        ):
            next_ = self[0]["centroids"]["array"][centroid_index:centroid_index+np.uint64(1)]
            centroid_index += np.uint64(1)
        else:
            next_ = centroid.new(values[values_index], 1.0)
            values_index += np.uint64(1)

        next_sum = next_[0]["mean"] * next_[0]["weight"]
        weight_so_far += next_[0]["weight"]

        if weight_so_far <= q_limit_times_count:
            sums_to_merge += next_sum
            weights_to_merge += next_[0]["weight"]
        else:
            result[0]["sum"] += centroid.add(cur[0], sums_to_merge, weights_to_merge)
            sums_to_merge = np.float64(0.0)
            weights_to_merge = np.float64(0.0)
            centroids.push_back(compressed, cur[0])
            q_limit_times_count = k_to_q(k_limit, self[0]["max_size"]) * result[0]["count"]
            k_limit += 1
            cur = next_

    result[0]["sum"] += centroid.add(cur[0], sums_to_merge, weights_to_merge)
    centroids.push_back(compressed, cur[0])
    centroids.sort(compressed)
    result[0]["centroids"] = compressed
    self[0] = result[0]


@nb.njit(cache=True)
def quantile(self: np.ndarray[TDigest], q: np.float64) -> np.float64:
    self: TDigest = self[0]
    if self["centroids"]["size"] == 0:
        return np.float64(0.0)
    rank: np.float64 = q * self["count"]
    pos = np.uint64(0)
    if q > np.float64(0.5):
        if q >= np.float64(1.0):
            return self["max"]
        pos = np.uint64(0)
        t = self["count"]
        for i in range(self["centroids"]["size"] - 1, -1, -1):
            c = self["centroids"]["array"][i]
            t -= c["weight"]
            if rank >= t:
                pos = np.uint64(i)
                break
    else:
        if q <= np.float64(0.0):
            return self["min"]
        pos = self["centroids"]["size"] - np.uint64(1)
        t = np.float64(0.0)
        for i in range(self["centroids"]["size"]):
            if rank < t + self["centroids"]["array"][i]["weight"]:
                pos = np.uint64(i)
                break
            t += self["centroids"]["array"][i]["weight"]

    delta = np.float64(0.0)
    min_ = self["min"]
    max_ = self["max"]
    if self["centroids"]["size"] > 1:
        if pos == np.uint64(0):
            delta = self["centroids"]["array"][pos + np.uint64(1)]["mean"] - self["centroids"]["array"][pos]["mean"]
            max_ = self["centroids"]["array"][pos + np.uint64(1)]["mean"]
        elif pos == self["centroids"]["size"] - np.uint64(1):
            delta = self["centroids"]["array"][pos]["mean"] - self["centroids"]["array"][pos - np.uint64(1)]["mean"]
            min_ = self["centroids"]["array"][pos - np.uint64(1)]["mean"]
        else:
            delta = (self["centroids"]["array"][pos + np.uint64(1)]["mean"] - self["centroids"]["array"][pos - np.uint64(1)]["mean"]) / 2
            min_ = self["centroids"]["array"][pos - np.uint64(1)]["mean"]
            max_ = self["centroids"]["array"][pos + np.uint64(1)]["mean"]

    c = self["centroids"]["array"][pos]
    value = c["mean"] + ((rank - t) / c["weight"] - np.float64(0.5)) * delta
    return clip(value, min_, max_)


@nb.njit(cache=True)
def k_to_q(k: np.float64, d: np.float64) -> np.float64:
    k_div_d: np.float64 = k / d
    if k_div_d >= np.float64(0.5):
        base: np.float64 = np.float64(1.0) - k_div_d
        return np.float64(1.0) - np.float64(2.0) * base * base
    else:
        return np.float64(2.0) * k_div_d * k_div_d


@nb.njit(cache=True)
def clip(x: np.float64, min_: np.float64, max_: np.float64) -> np.float64:
    if x < min_:
        return min_
    elif x > max_:
        return max_
    else:
        return x