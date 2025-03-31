import numpy as np
from eaf import get_empirical_attainment_surface, EmpiricalAttainmentFuncPlot
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from typing import List, Dict, Optional
from pymoo.indicators.hv import HV


def get_solutions(data_source: str, n_var: int, corr: float, algorithm: str, population_size: int = 100, iterations: int = 1) -> np.ndarray:
    """
    Function to get the solutions from an algorithm at a specified correlation and decision variable value.

    Parameters:
    -----------
        data_source (str): path to the root directory of your data e.g, 'data'
        n_var (int): decision variable value
        corr (float): correlation value
        algorithm (str): algorithm chosen
        iterations (int): number of iterations to load

    Returns
    -----------
        np.ndarray: numpy array containing the solutions from the specified algorithm at the correlation and decision variable value
    """

    assert type(n_var) == int
    assert type(corr) == float
    assert type(iterations) == int

    file_path = f'{data_source}/n_var_{n_var}/corr_{corr}'
    solutions = np.zeros((iterations, population_size, 2))

    for i in range(iterations):
        res = np.load(f'{file_path}/{i}/{algorithm}/result.npz')
        F = res['F']

        # Ensure F has shape (100, 2) by padding or truncating
        F_padded = np.zeros((population_size, 2))  # Default padding with zeros
        min_rows = min(F.shape[0], population_size)  # Get the valid row count to copy
        F_padded[:min_rows] = F[:min_rows]  # Copy the available data

        solutions[i] = F_padded  # Assign to the solutions array

        res.close()

    return solutions


def get_hv_ref_point(data_source: str, n_vars: List[int], correlations: List[float], algorithms: Optional[List[str]] = ["NSGA2"], population_size: int = 100, iterations: int = 1) -> HV:
    """
    Ensuring that the reference point is larger than the maximum value of the Pareto front from all the fronts from all the algorithms

    Parameters:
    -----------
        n_vars (List[int]): list of decision variables
        correlations (List[float]): list of correlations
        iterations (int): Number of iterations performed

    Returns:
    -----------
        np.ndarray: reference point
    """
    assert type(n_vars) == type(correlations) == list

    all_solutions = []

    for n_var in n_vars:
        for corr in correlations:
            for algorithm in algorithms:
                sols = get_solutions(data_source, n_var, corr, algorithm, population_size, iterations)
                for i in sols:
                    all_solutions.extend(i)

    max_x = max(np.array(all_solutions)[:, 0])
    max_y = max(np.array(all_solutions)[:, 1])

    ref_point = np.array([max_x * 1.1, max_y * 1.1])
    hv = HV(ref_point=ref_point)

    return hv

def get_callback(data_source: str, n_var: int, corr: float, algorithm: str, iterations: int = 1) -> List[Dict]:
    """
    Function to get the callback data from the algorithm at a specified correlation and decision variable value.

    Parameters:
    -----------
        data_source (str): path to the root directory of your data e.g, 'data'\
        n_var (int): decision variable value
        corr (float): correlation value
        algorithm (str): algorithm chosen
        iterations (int): number of iterations to load

    Returns:
    ----------
        List[Dict]: list of dictionaries containing the callback data from the specified algorithm at the correlation and decision variable value
    """
    assert type(data_source) == str
    assert type(n_var) == int
    assert type(corr) == float

    callbacks = []
    file_path = f'{data_source}/n_var_{n_var}/corr_{corr}'

    for i in range(iterations):
        callback = np.load(f'{file_path}/{i}/{algorithm}/callback.npz', allow_pickle=True)
        n_evals = callback['n_evals']
        opt_F = callback['opt_F']
        opt_X = callback['opt_X']
        callbacks.append({'n_evals': n_evals, 'opt_F': opt_F, 'opt_X': opt_X})
        callback.close()

    return callbacks


def plot_eaf(data: np.ndarray, name: str, nvar: int, corr: float, path: Optional[str] = None) -> None:
    """
    Function to plot the empirical attainment function (EAF) for a given set of data

    Parameters:
    -----------
        data (np.ndarray): numpy array containing the solutions
        name (str): name of the algorithm
        nvar (int): number of decision variables
        corr (float): correlation value
        path (str, optional): path to save the plot. Defaults to None.

    Returns:
    -----------
        None: The function saves the plot to the specified path if provided.
    """

    levels = [1, len(data) // 2, len(data)]
    surfs = get_empirical_attainment_surface(costs=data, levels=levels)

    _, ax = plt.subplots()
    eaf_plot = EmpiricalAttainmentFuncPlot()
    eaf_plot.plot_surface_with_band(ax, label=name, color='red', surfs=surfs)
    ax.legend()
    ax.grid()
    ax.set_xlabel("Objective 1")
    ax.set_ylabel("Objective 2")
    ax.set_title(f"EAF - {nvar} Decision Variables - Correlation: {corr}")

    if path is not None:
        plt.savefig(f'{path}/{corr}.pdf', bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_multiple_eafs(data: np.ndarray, names: List[str], nvar: int, corr: float, path: Optional[str] = None) -> None:
    """
    Function to plot multiple eafs from multiple different algorithms on the same problem

    Parameters:
    ------------
        data (np.ndarray): numpy array containing the solutions shaoe (num_algorithms, iterations, pop_size, 2)
        names (str): names of the algorithms
        nvar (int): number of decision variables
        corr (float): correlation value
        path (str, optional): path to save the plot. Defaults to None.

    Returns:
    -----------
        None: The function saves the plot to the specified path if provided.
    """

    num_eafs = data.shape[0]
    cols = ['red', 'green', 'blue', 'yellow', 'purple'] # Add more colours for more algorithms

    _, ax = plt.subplots()

    for i in range(num_eafs):
        levels = [1, len(data[i]) // 2, len(data[i])]
        surfs = get_empirical_attainment_surface(costs=data[i], levels=levels)

        eaf_plot = EmpiricalAttainmentFuncPlot()
        eaf_plot.plot_surface_with_band(ax, label=names[i], color=cols[i], surfs=surfs)

    ax.legend()
    ax.grid()
    ax.set_xlabel("Objective 1")
    ax.set_ylabel("Objective 2")
    ax.set_title(f"EAF - {nvar} Decision Variables - Correlation: {corr}")
    if path is not None:
        plt.savefig(f'{path}/{corr}.pdf', bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_convergence(data_source: str, hv: HV, n_var: int, corr: float, iterations: int = 1, algorithms: Optional[List[str]] = ["NSGA2"], save_data: bool = True, path: Optional[str] = None) -> None:
    """
    Function to plot the interquartile range of convergence of the algorithms for a given correlation and decision variable value.

    Note: This may take some time due to the computation of many hypervolumes.

    Parameters:
    -----------
        data_source (str): path to the root directory of your data e.g, 'data'
        hv (HV): hypervolume object
        n_var (int): decision variable value
        corr (float): correlation value
        iterations (int): number of iterations to load
        algorithms (List[str], optional): list of algorithms to plot. Defaults to ["NSGA2"].
        path (str, optional): path to save the plot. Defaults to None.

    Returns:
    -----------
        None: The function saves the plot to the specified path or shows it.
    """


    if type(algorithms) == str:
        algorithms = [algorithms]

    with (tqdm(total=iterations*len(algorithms)) as pbar):
        for i, algorithm in enumerate(algorithms):
            callbacks = get_callback(data_source, n_var, corr, algorithm, iterations)
            n_evals = callbacks[0]['n_evals']
            hypervolumes = []
            pbar.set_description(f"Decision Variables: {n_var}: Correlation: {corr}: Algorithm: {algorithm} ", refresh=True)

            for iter, callback in enumerate(callbacks):
                all_fronts = []
                opts = []
                for i, opt in enumerate(callback['opt_F']):
                    all_fronts.append(opt)
                    opts.extend([hv(np.array(all_fronts))])
                hypervolumes.append(opts)
                pbar.update(1)
            hypervolumes = np.array(hypervolumes)

            # Get median and quartiles
            q25, med, q75 = np.percentile(hypervolumes, [25 ,50, 75], axis=0)

            if algorithm == "NSGA2": algorithm="NSGA-II"  # Changing the name for the plot
            plt.plot(n_evals, med, label=algorithm)
            plt.fill_between(n_evals, q25, q75, alpha=0.3)

            if save_data:
                filepath = f'./plots/hypervolume_data/n_var_{n_var}/corr_{corr}'
                os.makedirs(filepath, exist_ok=True)
                np.savez(f'{filepath}/{algorithm}.npz', n_evals=n_evals, med=med, q25=q25, q75=q75)

    plt.xlabel("Number of Evaluations")
    plt.ylabel("Hypervolume")
    plt.legend()
    plt.title(f"Convergence Plot - {n_var} Decision Variables - Correlation: {corr}")

    if path is not None:
        os.makedirs(path, exist_ok=True)
        plt.savefig(f'./{path}/n_var_{n_var}/{corr}.pdf', bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_convergence_from_hypervolume_data(data_source: str, n_var: int, corr: float, algorithms: Optional[List[str]] = ["NSGA2"], path=None) -> None:
    """
    Function to plot the convergence from hypervolume data saved in npz files.

    Parameters:
    -----------
        data_source (str): path to the root directory of your data e.g, 'data'
        nvar (int): decision variable value
        corr (float): correlation value
        algorithms (List[str]): list of algorithms to plot
        path (str, optional): path to save the plot. Defaults to None.
    Returns:
    -----------
        None: The function saves the plot to the specified path or shows it.
    """

    for i, algorithm in enumerate(algorithms):
        path = f'{data_source}/n_var_{n_var}/corr_{corr}/{algorithm}.npz'
        data = np.load(path)
        n_evals = data['n_evals']
        med = data['med']
        q25 = data['q25']
        q75 = data['q75']

        plt.plot(n_evals, med, label=algorithm)
        plt.fill_between(n_evals, q25, q75, alpha=0.3)

    plt.xlabel("Number of Evaluations")
    plt.ylabel("Hypervolume")
    plt.legend()
    plt.title(f"Convergence Plot - {n_var} Decision Variables - Correlation: {corr}")

    if path is not None:
        os.makedirs(f'{path}/n_var_{n_var}', exist_ok=True)
        plt.savefig(f'{path}/n_var_{n_var}/{corr}.pdf', bbox_inches='tight')
    else:
        plt.show()

    plt.close()
