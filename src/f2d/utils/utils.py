"""Generic utils."""
import itertools
import multiprocessing as mp
import os
import time

import numpy as np
import tqdm
from pathos import multiprocessing as mp_


def get_randomizer(seed):
    return np.random.RandomState(seed=seed)


def get_seed(randomizer_or_none):
    if randomizer_or_none is None:
        return None
    return randomizer_or_none.randint(np.iinfo(np.uint32).max)


# Function to get a single list combining all list values in a dictionary.
def get_values(dict_of_lists):
    return list(itertools.chain(*dict_of_lists.values()))


def parallel_map(func, list_of_args, nprocs=None, chunksize=100, progress=True, use_pathos=True):
    """Executes function for list of args in parallel across multiple processes.

    Args:
      func: the function to execute, with signature func(a, b, c, ..).
      list_of_args: list of arguments. E.g. [(a0, b0, c0), (a1, b1, c1), ..].
      nprocs: number of processes to spawn. Defaults to `os.cpu_count()`.
      chunksize: number of chunks to pass to each process at once.
      progres: whether to show a tqdm-style progress bar.
      use_pathos: whether to use the `pathos` library for multiprocessing.

    Returns:
      result: list of returned values from func, in order of `list_of_args`.
    """
    if nprocs is None:
        nprocs = os.cpu_count()
    if progress:
        list_of_args = tqdm.tqdm(list_of_args)
    mp_lib = mp_ if use_pathos else mp
    with mp_lib.Pool(nprocs) as pool:
        # Note: consider using `imap` if there are any memory issues.
        result = pool.starmap(func, list_of_args, chunksize=chunksize)
        return result


if __name__ == "__main__":
    # Testing
    N = int(7e4)
    args_list = list(zip(range(2, N), range(2, N)))
    start_time = time.time()

    def is_prime(x, useless_y):
        for i in range(2, x):
            if x % i == 0:
                return x, False, useless_y
        return x, True, useless_y

    parallel_map(is_prime, args_list, progress=True)
    end_time = time.time()
    print("Elapsed time: {:.2f}".format(end_time - start_time))
