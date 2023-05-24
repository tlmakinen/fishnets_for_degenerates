import sympy
import numpy as np
from matplotlib import pyplot as plt
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import pysr

def load_data_pysr(fname, eta_component=0, n_params=6, skip=5,  install=True):


    data = np.load(fname)

    X = data[::skip, :n_params]
    y = data[::skip, n_params:]

    X = pd.DataFrame(X, columns=["A_", "B_", "C_", "D_", "E_", "F_"])

    return X, y
