from json import JSONDecodeError
import itertools
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import linregress, median_abs_deviation
import unionfind


pysat_solvers = ["cadical", "glucose3", "glucose4", "gluecard3", "gluecard4", "lingeling", "maple_chrono", "maple_cm", "maplesat", "mergesat3", "minicard", "minisat22", "minisatgh"]


def exp_linear_fit(exp, df, y_column_name, x_column_name="n", estimate_error=False, estimate_error_num_samples=1000, agg_name="mean"):
    """
    Perform an exponential fit in base 2 or a linear fit
    """
    agg_df = df.groupby(x_column_name).agg({y_column_name: ["mean", "median", "std", median_abs_deviation]})
    linreg_result = linregress(agg_df.index, np.log2(agg_df[y_column_name][agg_name]) if exp else agg_df[y_column_name][agg_name])
    if not estimate_error:
        return linreg_result
    linreg_slopes = []
    for resampling in range(estimate_error_num_samples):
        df_resample = df.sample(len(df) // 2)
        df_resample_agg = df_resample.groupby(x_column_name).agg({y_column_name: ["mean", "median", "std", median_abs_deviation]})
        linreg_result_resample = linregress(df_resample_agg.index, np.log2(df_resample_agg[y_column_name][agg_name]) if exp else df_resample_agg[y_column_name][agg])
        linreg_slopes.append(linreg_result_resample.slope)
    return linreg_result, np.std(linreg_slopes)


def exp_fit(df, y_column_name, x_column_name="n", estimate_error=False, estimate_error_num_samples=1000, agg_name="mean"):
    """
    Perform an exponential fit in base 2
    """
    return exp_linear_fit(True, df, y_column_name, x_column_name, estimate_error, estimate_error_num_samples, agg_name)


def linear_fit(df, y_column_name, x_column_name="n", estimate_error=False, estimate_error_num_samples=1000, agg_name="mean"):
    """
    Perform an exponential fit in base 2
    """
    return exp_linear_fit(False, df, y_column_name, x_column_name, estimate_error, estimate_error_num_samples, agg_name)


def clustering_analysis(instance_info, num_satisfied_lb):
    n, num_satisfied, num_clauses = instance_info["n"], np.array(instance_info["num_satisfied"]), len(instance_info["clauses_vars"])
    satisfying_assignments = np.argwhere(num_satisfied >= num_satisfied_lb).reshape(-1)
    distances_array = hamming_weight(satisfying_assignments[:, np.newaxis] ^ satisfying_assignments[np.newaxis, :], n)
    distances_dict = {(x0, x1): distances_array[i0, i1] for i0, x0 in enumerate(satisfying_assignments) for i1, x1 in enumerate(satisfying_assignments)}
    satisfying_assignments = sorted(satisfying_assignments, key=lambda assignment: num_satisfied[assignment])
    results = np.zeros((num_clauses - num_satisfied_lb + 1, n + 1, 5))
    for min_num_satisfied in range(num_satisfied_lb, num_clauses + 1):
        # Remove assignments with not enough satisfied clauses
        while satisfying_assignments and num_satisfied[satisfying_assignments[0]] < min_num_satisfied:
            satisfying_assignments.pop(0)
        if not satisfying_assignments:
            break
        # Compute number of clusters, minimal, maximal, mean and median cluster size for increasing clustering distance
        uf = unionfind.UnionFind(satisfying_assignments)
        distances_sorted_pairs = sorted(itertools.combinations(satisfying_assignments, 2), key=lambda pair: distances_dict[pair])
        for clustering_distance in range(0, n + 1):
            while distances_sorted_pairs and distances_dict[distances_sorted_pairs[0]] <= clustering_distance:
                uf.union(*distances_sorted_pairs[0])
                distances_sorted_pairs.pop(0)
            components_sizes = [len(component) for component in uf.components()]
            results[min_num_satisfied - num_satisfied_lb, clustering_distance, :] = [
                len(components_sizes),
                np.min(components_sizes),
                np.max(components_sizes),
                np.mean(components_sizes),
                np.median(components_sizes)
            ]
    return results


def hamming_weight(x, num_bits):
    return np.sum([(x >> j) & 1 for j in range(num_bits)], axis=0)


# From https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform
def inplace_fwht(a) -> None:
    """In-place Fast Walsh–Hadamard Transform of array a."""
    h = 1
    while h < a.shape[-1]:
        for i in range(0, a.shape[-1], h * 2):
            for j in range(i, i + h):
                x = np.copy(a[..., j])
                y = np.copy(a[..., j + h])
                a[..., j] = x + y
                a[..., j + h] = x - y
        h *= 2


def load_qaoa_benchmarks(directory, pattern="instance_benchmark_qaoa_result*.json", default_p=7):
    qaoa_benchmarks = []
    for result_idx, qaoa_result_file in enumerate(Path(directory).rglob(pattern)):
        if result_idx % 1000 == 0:
            logging.info(f"result {result_idx}")
        with qaoa_result_file.open("r") as f:
            try:
                result = json.load(f)
            except JSONDecodeError as e:
                #print(f"json decode error in {qaoa_result_file.resolve()}")
                continue
        if "p" not in result:
            result["p"] = default_p
        result["eval_qaoa_runtime"] = 1 / result["eval_qaoa_success_probability"]
        if "opt_eval_qaoa_success_probability" in result:
            result["opt_eval_qaoa_runtime"] = 1 / result["opt_eval_qaoa_success_probability"]
        if "opt2_eval_qaoa_success_probability" in result:
            result["opt2_eval_qaoa_runtime"] = 1 / result["opt2_eval_qaoa_success_probability"]
        del result["betas"]
        del result["gammas"]
        qaoa_benchmarks.append(result)
    return pd.DataFrame(qaoa_benchmarks)


def load_schoning_benchmarks(directory, pattern="instance_benchmark_schoning_iterative*.json"):
    schoning_benchmarks = []
    for result_idx, schoning_result_file in enumerate(Path(directory).rglob(pattern)):
        if result_idx % 1000 == 0:
            logging.info(f"result {result_idx}")
        with schoning_result_file.open("r") as f:
            result = json.load(f)
        result["schoning_runtime"] = result["num_trials"]
        del result["num_trials"]
        del result["assignment"]
        schoning_benchmarks.append(result)
    return pd.DataFrame(schoning_benchmarks)


def load_schoning_qaoa_benchmarks(directory, pattern="instance_benchmark_qaoa_schoning*.json"):
    schoning_qaoa_benchmarks = []
    for result_idx, schoning_qaoa_result_file in enumerate(Path(directory).rglob(pattern)):
        if result_idx % 1000 == 0:
            logging.info(f"result {result_idx}")
        with schoning_qaoa_result_file.open("r") as f:
            result = json.load(f)
        result["schoning_qaoa_runtime"] = result["num_trials"]
        del result["num_trials"]
        del result["assignment"]
        schoning_qaoa_benchmarks.append(result)
    return pd.DataFrame(schoning_qaoa_benchmarks)


def load_walksatlm_benchmarks(directory, pattern="instance_benchmark_walksat_lm*.json"):
    walksatlm_benchmarks = []
    for result_idx, walksatlm_result_file in enumerate(Path(directory).rglob(pattern)):
        if result_idx % 1000 == 0:
            logging.info(f"result {result_idx}")
        with walksatlm_result_file.open("r") as f:
            try:
                result = json.load(f)
            except JSONDecodeError as e:
                print(f"json decode error in {walksatlm_result_file.resolve()}")
                continue
        if "num_trials" in result:
            result["walksatlm_runtime"] = result["num_trials"]
            del result["num_trials"]
            del result["assignment"]
        elif "num_successes" in result:
            pass
        else:
            raise ValueError("Walksatlm benchmark result should include either 'num_trials' or 'num_successes'")
        walksatlm_benchmarks.append(result)
    return pd.DataFrame(walksatlm_benchmarks)


def load_pysat_benchmarks(directory, pattern="instance_benchmark_pysat*.json"):
    pysat_benchmarks = []
    for result_idx, pysat_result_file in enumerate(Path(directory).rglob(pattern)):
        if result_idx % 1000 == 0:
            logging.info(f"result {result_idx}")
        with pysat_result_file.open("r") as f:
            result = json.load(f)
            pysat_benchmarks.append(result)
    pysat_benchmarks_df = pd.DataFrame(pysat_benchmarks)
    for solver_name in pysat_solvers:
        pysat_benchmarks_df[solver_name + "_runtime"] = pysat_benchmarks_df[solver_name + "_decisions"] + pysat_benchmarks_df[solver_name + "_propagations"]
    return pysat_benchmarks_df


def load_instances_features(directory, pattern="*features*.json"):
    instances_features = []
    for idx, instance_features_file in enumerate(Path(directory).rglob(pattern)):
        logging.info(f"instance feature file {idx}")
        with instance_features_file.open("r") as f:
            instance_features = json.load(f)
        instances_features.append(instance_features)
    return pd.DataFrame(instances_features)
