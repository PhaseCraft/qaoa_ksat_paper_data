import numpy as np
from scipy.special import binom


# Credit: https://stackoverflow.com/questions/46374185/does-python-have-a-function-which-computes-multinomial-coefficients
def multinomial(params):
    if len(params) == 1:
        return 1
    return binom(sum(params), params[-1]) * multinomial(params[:-1])


def multinomial_partitions(num_vars, s):
    if num_vars == 1:
        yield (s,)
        return
    for sp in range(s + 1):
        for first_vars in multinomial_partitions(num_vars - 1, s - sp):
            yield first_vars + (sp,)


def multinomial_sum(n, f, *args):
    return sum([multinomial(variables) * np.product(np.array(args) ** np.array(variables)) * f(*variables) for variables in multinomial_partitions(len(args), n)])
