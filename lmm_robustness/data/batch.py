from typing import Iterable


def batch(d: Iterable, batch_size: int) -> Iterable:
    """ e.g. [(x, y), (x, y), ...] -> [([x, x, ...], [y, y, ...]), ...] """
    batched_data = None
    for data_tuple in d:  # e.g. (x, y)
        if batched_data is None:
            batched_data = [[x] for x in data_tuple]
        else:
            assert len(data_tuple) == len(batched_data)
            for i, x in enumerate(data_tuple):
                batched_data[i].append(x)

        if len(batched_data[0]) == batch_size:
            yield batched_data
            batched_data = None
