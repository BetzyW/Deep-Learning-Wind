"""Comprueba que entrenar los pliegues juntos da lo mismo que entrenar cada pliegue por separado (sin Dropout, para
que sea determinista) y que los huecos se excluyen. Datos sintéticos, CPU, segundos."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from timeseries_cv import Lotes, entrenar_configuracion, glorot, problema_cv, problema_holdout  # noqa: E402

CFG = dict(dropout=0.5, epocas=4, lote=32, tasa_aprendizaje=0.001)


def serie(n=1500, huecos=True):
    t = np.arange(n)
    v = 4 + np.sin(2 * np.pi * t / 24) + 0.3 * np.random.default_rng(0).normal(size=n)
    if huecos:
        v[600:650] = np.nan
    return v


def test_lotes_equivale_a_pliegues_separados():
    torch.manual_seed(0)
    v = serie()
    I, O, J = 48, 24, 24
    P = problema_cv(v, 1300, I, O, J, "cpu")
    assert P.descartados > 0                                       # los pliegues que tocan el hueco se descartan
    capas, mascara = (16, 8), (0, 0)
    dims = [I, *capas, O]
    gen = torch.Generator().manual_seed(1)
    ini = [glorot(P.F, a, b, gen, "cpu") for a, b in zip(dims[:-1], dims[1:])]
    _, _, juntos = entrenar_configuracion(P, capas, mascara, CFG, 0, "cpu", iniciales=ini)
    X, Y = P.Xg.numpy(), P.Yg.numpy()
    for f in [0, 3, P.F // 2, P.F - 1]:
        uno = Lotes(X, Y, np.array([P.m[f]]), P.Xcv_crudo[f:f + 1], P.Ycv_crudo[f:f + 1], "cpu")
        _, _, solo = entrenar_configuracion(uno, capas, mascara, CFG, 0, "cpu", iniciales=[w[f:f + 1] for w in ini])
        np.testing.assert_allclose(juntos[:, f], solo[:, 0], rtol=1e-4, atol=1e-5)


def test_holdout_no_usa_objetivos_del_holdout():
    v = serie(huecos=False)
    H = problema_holdout(v, 1300, 96, 24, 24, "cpu")
    assert (int(H.m[0]) - 1) * 24 + 96 + 24 <= 1300                # el último objetivo de entrenamiento queda antes del holdout
    assert H.Xcv.shape[1] > 0
