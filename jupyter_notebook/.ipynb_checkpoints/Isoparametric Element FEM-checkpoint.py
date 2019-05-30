# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd


# E: Young's module PR: Poisson's ratio
def elastic_matrix(E, PR):
    return (E / (1 - PR ** 2)) * np.array([[1, PR, 0], [PR, 1, 0], [0, 0, (1 - PR) / 2]])


# +
# ex) nodes = np.array([[1, 1], [1, 0], [0, 0]])

def shape_fucntions(nodes):
    DSF = np.hstack([np.array([[1], [1], [1]]), nodes])
    N1 = np.linalg.solve(DSF, np.array([1, 0, 0]))
    N2 = np.linalg.solve(DSF, np.array([0, 1, 0]))
    N3 = np.linalg.solve(DSF, np.array([0, 0, 1]))
    N = np.array([N1, N2, N3])
    return N
# -


