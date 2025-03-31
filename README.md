# BEACON Problem Optimisation

This project provides a framework for optimizing BEACON problems using the `pymoo` library. It includes classes for defining the problem, tracking convergence, and running the optimization.

## Requirements

- Python
- pip

## Installation

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

The **pymoo** package may need compilation for significant performance improvements. If you encounter issues, consider installing the package from source. And see the [pymoo documentation](https://github.com/anyoptimization/pymoo) for more details.

## Usage

### BEACON Class

The `BEACON` class is used to define the problem parameters and load/save problem configurations.

### BEACONProblem Class

The `BEACONProblem` class encapsulates the BEACON problem into a `pymoo` problem.

### BEACONProblemOptimiser Class

The `BEACONProblemOptimiser` class is used to optimize a BEACON problem using a specified `pymoo` algorithm.

### Example

```python
import torch
from pymoo.algorithms.moo.nsga2 import NSGA2
from BEACON import *

lb, ub = 0., 1.

# Define the BEACON problem, example initialisation
beacon = BEACON(
    num_features = 1000,
    input_dim = 2,
    lengthscale = 0.01,
    correlation = 0.5,
)

# Save the problem to a file if needed
beacon.save_problem(lb=lb, ub=ub)

# Load problem from file if you have a pre-defined problem
beacon.load_problem_from_file('path/to/problem.npz')

# Define the BEACON problem for pymoo
problem = BEACONProblem(beacon_sampler=beacon, xl=lb, xu=ub)

# Define the optimization algorithm, e.g., NSGA2
algorithm = NSGA2(pop_size=100)

# Create the optimizer
optimizer = BEACONProblemOptimiser(problem=problem, algorithm=algorithm, lb=lb, ub=ub, iteration=0)

# Run the optimization
result = optimizer.minimise_correlated_problem(n_gen=100, save_history=True)
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.