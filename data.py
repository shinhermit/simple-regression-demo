import csv
from typing import (List, Union)
from numbers import Number


class Vector:
    """
    Represents a mathematical vector.

    Supports vector operations:
    - addition,
    - substraction
    - dot product
    - scaling (multiplication by a scalar)

    ex:
    >>> print(2 * Vector(1,1) + Vector(1,2))
    [3, 4]
    """

    def __init__(self, *elts: float):
        self.elts = [e for e in elts]

    def __str__(self) -> str:
        return str(self.elts)

    def __iter__(self) -> 'Vector':
        self.iterator = iter(self.elts)
        return self

    def __next__(self) -> float:
        return next(self.iterator)

    def __getitem__(self, i: int) -> float:
        return self.elts[i]

    def __len__(self) -> int:
        return len(self.elts)

    def __add__(self, other: 'Vector') -> 'Vector':
        vec_sum = self.__class__()
        for i in range(min(len(self), len(other))):
            vec_sum.append(self[i] + other[i])
        return vec_sum

    def __sub__(self, other: 'Vector') -> 'Vector':
        vec_sum = self.__class__()
        for i in range(min(len(self), len(other))):
            vec_sum.append(self[i] - other[i])
        return vec_sum

    def __mul__(self, other: Union[float, 'Vector']) -> Union[float, 'Vector']:
        if isinstance(other, Number):
            vec_scaled = self.__class__()
            for i in range(len(self)):
                vec_scaled.append(self[i] * other)
            return vec_scaled
        else:
            dot_prod = 0
            for i in range(min(len(self), len(other))):
                dot_prod += self[i] * other[i]
            return dot_prod

    def __rmul__(self, other: Union[float, 'Vector']) -> Union[float, 'Vector']:
        return self.__mul__(other)

    def append(self, elem: float):
        self.elts.append(elem)


class Experiment:
    """
    Represents an experiment, aka row of a dataset.

    It has a nuber of inputs and 1 output.
    """

    def __init__(self, inputs: Vector = Vector(), output: float = 0):
        self.inputs: Vector = inputs
        self.output: float = output


class Dataset:
    """
    Represents a dataset as a collection of observations (experiments).
    """

    features: list = []
    experiments: List[Experiment] = []

    def __iter__(self) -> 'Dataset':
        self.iterator = iter(self.experiments)
        return self

    def __next__(self) -> Experiment:
        return next(self.iterator)

    def __getitem__(self, i: int) -> Experiment:
        return self.experiments[i]

    @staticmethod
    def from_csv(csv_str: str) -> 'Dataset':
        dataset = Dataset()
        doc = csv.reader(csv_str.split("\n"))
        dataset.features = list(next(doc)[:-1])
        for row in doc:
            inputs = Vector(*[float(v) for v in row[:-1]])
            output = float(row[-1])
            dataset.experiments.append(
                Experiment(inputs, output))
        return dataset


if __name__ == "__main__":
    print(2 * Vector(1, 1) + Vector(1, 2)) # expected: [3, 4]
    print(Vector(1, 1) * Vector(1, 2)) # expected: 3
    print(Vector(1, 1) - Vector(1, 2)) # expected: [0, -1]
