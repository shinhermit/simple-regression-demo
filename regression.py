from data import (Vector, Dataset)
from typing import Tuple


def sqr(x: float) -> float:
    return x*x


def sqr_error(dataset: Dataset, rho: Vector) -> float:
    error = 0.
    for experiment in dataset:
        error += sqr(experiment.output - rho * experiment.inputs)
    return error / 2


def gradient(dataset: Dataset, rho: Vector) -> Vector:
    grad = Vector()
    for i in range(len(dataset.features)):
        part_deriv = 0
        for experiment in dataset:
            part_deriv += (experiment.output - rho * experiment.inputs) * experiment.inputs[i]
        grad.append(-part_deriv)
    return grad


def fit_linear(dataset: Dataset, lambdaa=0.1,
               max_iter=10000, threshold=0.11) -> Tuple[Vector, float, int]:
    rho = Vector(*[i for i in range(len(dataset.features))])
    error = sqr_error(dataset, rho)
    prev_error = 0
    iter_count = 0
    while iter_count < max_iter and (prev_error == 0 or abs(prev_error - error) > threshold) :
        rho = rho - lambdaa * gradient(dataset, rho)
        prev_error, error = error, sqr_error(dataset, rho)
        iter_count += 1
    return rho, error, iter_count


if __name__ == "__main__":
    csv_str = """V_lead,V_iron,V_aluminium,mass
0.3,0.2,0.1,5.246
0.1,0.1,0.4,3.001
0.7,0.3,0.5,11.649
0.4,0.6,0.11,9.5574"""

    rho, sqr_err, nb_iter = fit_linear(Dataset.from_csv(csv_str))

    print("Densities:", rho)
    print("Nb iterations:", nb_iter)
    print("Square error:", sqr_err)
