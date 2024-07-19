import numpy as np


def biased_coin_toss(n, p=0.25):
    results = np.random.binomial(1, p, n)
    count_of_ones = np.sum(results)
    return count_of_ones
