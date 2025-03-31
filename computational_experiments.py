from BEACON import *

from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.algorithms.moo import nsga2, sms, moead
from pymoo.core.evaluator import Evaluator
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.util.ref_dirs import get_reference_directions

from tqdm import tqdm
from typing import List


def generate_beacon_problems(n_vars: List[int], correlation_values: List[float], num_iterations: int, num_features: int, output_dim: int, lengthscale: float, lb: float, ub: float):
    for n_var in n_vars:
        for correlation in correlation_values:
            for i in range(num_iterations):
                correlated_problem = BEACON(
                    num_features=num_features,
                    input_dim=n_var,
                    output_dim=output_dim,
                    lengthscale=lengthscale,
                    correlation=correlation,
                    iteration=i
                )

                correlated_problem.save_problem(lb=lb, ub=ub)

    print("Problems generated")


def run_computational_experiments(n_vars: List[int], correlation_values: List[float], lb: float, ub: float, num_iterations: int):

    pop_size = 100
    n_gen = 1000

    # Parameters for the algorithms
    crossover_eta = 20
    crossover_prob = 0.9
    mutation_eta = 20
    mutation_prob = 0.1


    # Optimise Correlated Problems
    for n_var in n_vars:
        for correlation in correlation_values:
            with (tqdm(total=num_iterations) as pbar):
                for i in range(num_iterations):
                    pbar.set_description(f"Decision Vars: {n_var} - Correlation: {correlation} - Iteration: {i} / {num_iterations}")

                    # Load the BEACON problem
                    file_path = f"data/n_var_{n_var}/corr_{correlation}/{i}/problem.npz"
                    correlated_problem_sampler = BEACON()
                    correlated_problem_sampler.load_problem_from_file(file_path)

                    # Initialise the problem as a pymoo problem instance
                    problem = BEACONProblem(correlated_problem_sampler, xl=lb, xu=ub)

                    # Generate initial points to keep the same for all the algorithms
                    sampling = FloatRandomSampling()
                    pop = sampling(problem, pop_size)
                    Evaluator().eval(problem, pop)

                    # Generate reference directions for MOEAD
                    reference_directions = get_reference_directions('uniform', n_dim=correlated_problem_sampler.output_dim, n_points=pop_size)


                    # Initialise the algorithms
                    algorithmNSGA2 = nsga2.NSGA2(pop_size=pop_size,
                                                 sampling=pop,
                                                 crossover=SBX(eta=crossover_eta, prob=crossover_prob),
                                                 mutation=PolynomialMutation(eta=mutation_eta, prob=mutation_prob)

                                                 )

                    algorithmSMSEMOA = sms.SMSEMOA(pop_size=pop_size,
                                                   sampling=pop,
                                                   crossover=SBX(eta=crossover_eta, prob=crossover_prob),
                                                   mutation=PolynomialMutation(eta=mutation_eta, prob=mutation_prob)
                                                   )

                    algorithmMOEAD = moead.MOEAD(
                        ref_dirs=reference_directions,
                        sampling=pop,
                        crossover=SBX(eta=crossover_eta, prob_var=crossover_prob),
                        mutation=PolynomialMutation(eta=mutation_eta, prob=mutation_prob)
                    )

                    algorithms = [algorithmNSGA2, algorithmSMSEMOA, algorithmMOEAD]

                    # Optimise the problem with each algorithm
                    for algorithm in algorithms:
                        pbar.set_postfix_str(f"Algorithm: {algorithm.__class__.__name__}", refresh=True)
                        optimisation = BEACONProblemOptimiser(problem, algorithm, lb=lb, ub=ub, iteration=i)
                        res = optimisation.minimise_correlated_problem(n_gen=n_gen, save_history=False)

                    pbar.update()


if __name__ == "__main__":
    output_dim = 2
    num_features = 1000
    lengthscale = 0.01
    lb = 0.
    ub = 1.

    correlation_values = [1., .75, .5, .25, 0., -.25, -.5, -.75, -1.]
    n_vars = [1, 5, 10, 20]

    num_iterations = 20

    generate_beacon_problems(n_vars, correlation_values, num_iterations, num_features, output_dim, lengthscale, lb, ub)
    run_computational_experiments(n_vars, correlation_values, lb, ub, num_iterations)