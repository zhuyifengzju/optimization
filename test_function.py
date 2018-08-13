"""Test script for function implementation."""

import pytest
from function.rosenbrock import RosenbrockFunction
import numpy as np

def test_rosenbrock_value_2_dim():
    f = RosenbrockFunction().value
    g = RosenbrockFunction().grad_value

    # Test 2 dimension case optimal point
    x = np.array([1, 1])
    assert f(x) == 0
    assert np.count_nonzero(g(x)) == 0

def test_rosenbrock_value_n_dim():
    f = RosenbrockFunction().value
    g = RosenbrockFunction().grad_value
    
    # Test n dimension case

    # n = 3, x = [1, 1, 1]
    x = np.array([1, 1, 1])
    assert f(x) == 0
    assert np.count_nonzero(g(x)) == 0
    assert np.shape(g(x)) == (3,)

    # 4 <= n <= 7

    # x = [1, 1, ..., 1] -> global minimum
    x = np.array([1, 1, 1, 1])
    assert f(x) == 0
    assert np.count_nonzero(g(x)) == 0

    x = np.array([1, 1, 1, 1, 1, 1, 1])
    assert f(x) == 0
    assert np.count_nonzero(g(x)) == 0
    
