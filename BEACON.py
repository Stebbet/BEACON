import torch
import numpy as np

from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.core.callback import Callback

from typing import Optional, Dict, Any
import os



class BEACON:
    """
    BEACON: Continuous Bi-objective Benchmark problems with Explicit Adjustable COrrelatioN control

    This class generates a bi-objective multi-objective problem (MOP) using Random Fourier Features (RFF) of the RBF kernel where the user can explicitly specify the correlation between the two obejctives.

    Parameters:
    -------
        correlation (Optional[float]): The correlation between the objectives, between -1 and 1
        num_features (Optional[int]): number of features to use for the RFF approximation
        input_dim (Optional[int]): number of input dimensions
        lengthscale (Optional[float]): length-scale of the RBF kernel
        file_root (Optional[str]): Root directory to save the problems, defaults to 'data'
        iteration (int): Iteration of the problem to save, used for file naming when generating multiple problems
    """

    def __init__(
            self,
            correlation: Optional[float] = None,
            num_features: Optional[int] = None,
            input_dim: Optional[int] = None,
            lengthscale: Optional[float] = None,
            file_root: Optional[str] = 'data',
            iteration: int = 0
    ) -> None:
        """
        Initialise BEACON

        Parameters:
        -------
            correlation (Optional[float]): The correlation between the objectives, between -1 and 1
            num_features (Optional[int]): number of features to use for the RFF approximation
            input_dim (Optional[int]): number of input dimensions
            lengthscale (Optional[float]): length-scale of the RBF kernel
            file_root (Optional[str]): Root directory to save the problems, defaults to 'data'
            iteration (int): Iteration of the problem to save, used for file naming when generating multiple problems
        """

        assert correlation is None or -1 <= correlation <= 1, "Correlation must be between -1 and 1"

        self.num_features = num_features
        self.input_dim = input_dim
        self.output_dim = 2
        self.lengthscale = lengthscale
        self.correlation = correlation

        self.omegas: Optional[torch.Tensor] = None
        self.phi: Optional[torch.Tensor] = None
        self.weights: Optional[torch.Tensor] = None
        self.rff_scaling: Optional[torch.Tensor] = None
        self.mixing_matrix: Optional[torch.Tensor] = None
        self.iteration = iteration

        if any([correlation is not None, num_features is not None, input_dim is not None, lengthscale is not None]):

            # Cholesky decompose the mixing matrix
            self.mixing_matrix = torch.as_tensor(
                [[1, 0], [correlation, np.sqrt(1 - np.square(correlation))]], dtype=torch.float64
            )

            # Spectral Distribution of the RBF Kernel
            self.omegas = torch.as_tensor(np.random.normal(size=(self.num_features, self.input_dim))) / self.lengthscale

            self.phi = torch.rand(self.num_features, dtype=torch.float64) * 2 * np.pi
            self.weights = torch.randn(self.output_dim, self.num_features, dtype=torch.float64)
            self.rff_scaling = torch.sqrt(torch.tensor(2.0 / self.num_features, dtype=torch.float64))

        # Set the file path for saving problems
        self.file_root = file_root
        self.file_path = f'./{self.file_root}/n_var_{self.input_dim}/corr_{self.correlation}/{self.iteration}'


    def sample(self, x_targets: torch.Tensor) -> torch.Tensor:
        """
        Sample the correlated problem at given target points

        Parameters:
        -------
            x_targets (torch.Tensor): target points to sample at in decision space

        Returns:
        -------
            torch.Tensor: sampled values in objective space
        """

        Z_target = self.rff_scaling * torch.cos(torch.matmul(x_targets, self.omegas.T) + self.phi)

        # Apply the mixing matrix to the functions to correlate them
        output_at_target = Z_target @ self.weights.T @ self.mixing_matrix.T
        return output_at_target

    def load_problem(
            self,
            omegas: torch.Tensor,
            weights: torch.Tensor,
            phi: torch.Tensor,
            num_features: int,
            mixing_matrix: torch.Tensor,
            lengthscale: float,
            input_dim: int,
    ) -> None:
        """
        If you have the omega values already, you can load them into the class using this function.

        Parameters:
        -------
            omegas (torch.Tensor): omega values
            weights (torch.Tensor): weights
            phi (torch.Tensor): phi values
            num_features (int): number of features
            mixing_matrix (torch.Tensor): mixing matrix in its Cholesky decomposed form
            lengthscale (float): lengthscale
            input_dim (int): input dimension

        Returns:
        -------
            None
        """

        self.omegas = omegas
        self.weights = weights
        self.phi = phi
        self.num_features = num_features
        self.mixing_matrix = mixing_matrix
        self.lengthscale = lengthscale
        self.input_dim = input_dim

        self.rff_scaling = torch.sqrt(torch.tensor(2.0 / self.num_features, dtype=torch.float64))

    def load_problem_from_file(self, problem_filepath: str) -> None:
        """
        Load the problem from a file generated from the save_problem() function

        Parameters:
        -------
            problem_filepath (str): filepath to the problem file

        Returns:
        -------
            None
        """
        npzfile = np.load(problem_filepath)
        self.omegas = torch.as_tensor(npzfile['omegas'])
        self.weights = torch.as_tensor(npzfile['weights'])
        self.phi = torch.as_tensor(npzfile['phi'])
        self.num_features = int(npzfile['num_features'])
        self.mixing_matrix = torch.as_tensor(npzfile['mixing_matrix'])
        self.correlation = float(npzfile['correlation'])
        self.lengthscale = float(npzfile['lengthscale'])
        self.input_dim = int(npzfile['input_dim'])

        assert None not in [self.omegas, self.weights, self.phi, self.num_features, self.correlation,
                            self.mixing_matrix, self.lengthscale, self.input_dim]

        self.rff_scaling = torch.sqrt(torch.tensor(2.0 / self.num_features, dtype=torch.float64))
        npzfile.close()

    def save_problem(self, lb: float, ub: float) -> None:
        """
        Save the problem to a file

        Parameters:
        -------
            lb (float): lower bounds of the problem for reference when generating the problem
            ub (float): upper bounds of the problem for reference when generating the problem

        Returns:
        -------
            None
        """
        assert None not in [self.omegas, self.weights, self.phi, self.num_features, self.correlation,
                            self.mixing_matrix, self.lengthscale, self.input_dim, self.output_dim]

        os.makedirs(self.file_path, exist_ok=True)
        np.savez(f'{self.file_path}/problem.npz', omegas=self.omegas, weights=self.weights, phi=self.phi,
                 num_features=self.num_features, correlation=self.correlation, mixing_matrix=self.mixing_matrix,
                 lengthscale=self.lengthscale, input_dim=self.input_dim, output_dim=self.output_dim, lb=lb, ub=ub)


class BEACONProblem(Problem):
    """
    Problem class for pymoo - encapsulating the BEACON problem into a pymoo problem
    Parameters:
    -------
        beacon_sampler (BEACON): BEACON object to draw samples from
        xl (float): lower bounds of the problem
        xu (float): upper bounds of the problem
    """

    def __init__(self, beacon_sampler: BEACON, xl: float, xu: float) -> None:
        """
        Initialise the BEACONProblem
        Parameters:
        -------
            beacon_sampler (BEACON): BEACON object to draw samples from
            xl (float): lower bounds of the problem
            xu (float): upper bounds of the problem
        """

        assert isinstance(beacon_sampler, BEACON), "BEACONProblem must be initialised with an instance of BEACON"

        self.beacon_sampler = beacon_sampler
        n_var = self.beacon_sampler.input_dim
        n_obj = self.beacon_sampler.output_dim
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=0, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        # Ensure x is a 2D array (n_samples, n_var)
        assert x.shape[1] == self.n_var, f"Expected {self.n_var} vars, got {x.shape[1]}"

        samples = self.beacon_sampler.sample(torch.as_tensor(x))
        out["F"] = samples


class ConvergenceCallback(Callback):
    """
    Callback to track the convergence of the algorithm
    See https://pymoo.org/misc/convergence.html to learn more about the callbacks
    """

    def __init__(self) -> None:
        super().__init__()
        self.n_evals = []
        self.opt_F = []
        self.opt_X = []

    def notify(self, algorithm: Any):
        self.n_evals.append(algorithm.evaluator.n_eval)
        self.opt_F.append(algorithm.opt)
        self.opt_X.append(algorithm.opt)


class BEACONProblemOptimiser:
    """
    Optimise a BEACON problem using a given algorithms

    Parameters:
    -------
        problem (BEACONProblem): BEACON problem to optimise
        algorithm: pymoo algorithm to use for optimisation
        lb (float): lower bounds of the problem
        ub (float): upper bounds of the problem
        iteration (int): iteration number for saving results
    """

    def __init__(self, problem: BEACONProblem, algorithm: Any, lb: float = 0., ub: float = 1., iteration: int = 0) -> None:
        """
        Initialise the BEACONProblemOptimiser

        Parameters:
        -------
            problem (BEACONProblem): BEACON problem to optimise
            algorithm: pymoo algorithm to use for optimisation
            lb (float): lower bounds of the problem
            ub (float): upper bounds of the problem
            iteration (int): iteration number for saving results
        """


        self.problem = problem
        self.algorithm = algorithm

        self.history = None
        self.callback = ConvergenceCallback()
        self.lb = lb
        self.ub = ub
        self.file_path = f'./data/n_var_{problem.n_var}/corr_{problem.beacon_sampler.correlation}/{iteration}/{self.algorithm.__class__.__name__}'
        os.makedirs(self.file_path, exist_ok=True)

        # Ensure the problem is set up correctly
        assert isinstance(self.problem, BEACONProblem), "Problem must be an instance of BEACONProblem"

    def minimise_problem(self, n_gen, save_history=False):
        """
        Minimise the correlated problem using the given algorithm using pymoo

        Parameters:
        -------
            n_gen (int): number of generations to run the algorithm for
            save_history (bool): whether to save the history of the optimisation (WARNING: save_history=True can make the file sizes very large and may slow down the optimisation process)
        """

        res = minimize(self.problem,
                       self.algorithm,
                       termination=('n_gen', n_gen),
                       seed=1,
                       verbose=False,
                       save_history=save_history,
                       callback=self.callback
                       )

        if save_history:
            np.savez(f'{self.file_path}/history.npz', history=res.history)
            self.history = res.history

        # Save the final result and the callback
        np.savez(f'{self.file_path}/result.npz', X=res.X, F=res.F, G=res.G, CV=res.CV)
        np.savez(f'{self.file_path}/callback.npz', n_evals=self.callback.n_evals, opt_F=self.callback.opt_F,
                 opt_X=self.callback.opt_X)

        return res
