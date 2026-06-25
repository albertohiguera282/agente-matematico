# prueba_regresion.py

import numpy as np

def regresion_multiple(X, Y, Z):
    """
    Regresión múltiple:
    Z = a*X + b*Y + c
    """

    X = np.array(X, dtype=float)
    Y = np.array(Y, dtype=float)
    Z = np.array(Z, dtype=float)

    if not (len(X) == len(Y) == len(Z)):
        raise ValueError(
            "Las variables X, Y y Z deben tener la misma cantidad de datos."
        )

    A = np.vstack([
        X,
        Y,
        np.ones_like(X)
    ]).T

    modelo, residuos, rango, sing_vals = np.linalg.lstsq(
        A,
        Z,
        rcond=None
    )

    a, b, c = modelo

    Z_pred = a * X + b * Y + c

    ss_res = np.sum((Z - Z_pred) ** 2)
    ss_tot = np.sum((Z - np.mean(Z)) ** 2)

    r2 = 1 - (ss_res / ss_tot)

    return {
        "Coef_X": a,
        "Coef_Y": b,
        "Intercepto": c,
        "R2": r2,
        "Z_Predicho": Z_pred
    }

from scipy.stats import linregress

def regresion_simple(X, Y):

    if len(X) != len(Y):
        raise ValueError(
            "X e Y deben tener la misma cantidad de datos."
        )

    pendiente, intercepto, r, p, error = linregress(X, Y)

    return {
        "Pendiente": pendiente,
        "Intercepto": intercepto,
        "Correlacion": r,
        "R2": r**2,
        "P_Valor": p
    }