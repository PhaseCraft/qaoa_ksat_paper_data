#   Copyright 2023 Phasecraft Ltd.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License

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
