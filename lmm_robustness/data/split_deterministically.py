import numpy as np
from torch.utils.data import Subset


def split_deterministically(d, split_sizes: list[int], shuffle=True):
    assert sum(split_sizes) <= len(d), "Split sizes exceed dataset size."
    indices = [i for i in range(len(d))]

    np.random.seed(1111)
    if shuffle:
        np.random.shuffle(indices)

    subsets = []
    start = 0
    for size in split_sizes:
        subset_indices = indices[start:start + size]
        s = Subset(d, subset_indices)
        subsets.append(s)
        start += size

    return subsets
