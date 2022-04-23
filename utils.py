import numpy as np


def random_index_generator(count):
    indices = np.arange(0, count)
    np.random.shuffle(indices)

    for idx in indices:
        yield idx
