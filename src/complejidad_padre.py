"""Complejidad empírica de PADRE-TCN al variar el tamaño de la entrada y el ancho (sección 6.5 del enunciado del E3):
parámetros, FLOPs por ventana (contador de PyTorch) y tiempo de inferencia por ventana en la CPU del PC (4 hilos,
lote 1 024, mediana de 10 repeticiones), para contrastarlos con el análisis O(·) del capítulo de diseño.

  * Longitud de la ventana L ∈ {168, 240, 336, 504, 672} con el ancho de V6 (w = (64, 64, 16)) y K = 6.
    Predicción: FLOPs y tiempo lineales en L (pendiente ≈ 1 en escala log-log); parámetros constantes.
  * Ancho c ∈ {16, 32, 64, 128} con w = (c, c, c/4) y L = 168. Predicción: FLOPs y parámetros ≈ cuadráticos en c
    (pendiente ≈ 2), porque cada convolución cuesta k · c_entrada · c_salida por paso.
Los pesos son aleatorios (el costo no depende de sus valores). Salida: resultados/e3/complejidad_padre.csv
Uso: python src/complejidad_padre.py      (dl_env)
"""
import os, sys, time
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.dirname(__file__))
from original_model import PadreTCN  # noqa: E402

D = 4


def medir(L, w):
    m = PadreTCN(L, D, w=w, b=(1, 1, 0), K=6, lam=0.0).eval()
    params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    with torch.no_grad():
        with FlopCounterMode(display=False) as fc:
            m(torch.randn(1, L, D))
        x = torch.randn(1024, L, D)
        for _ in range(3):
            m(x)
        t = []
        for _ in range(10):
            t0 = time.perf_counter(); m(x); t.append(time.perf_counter() - t0)
    return dict(L=L, ancho=w[0], parametros=params, mflops_ventana=fc.get_total_flops() / 1e6,
                ms_ventana_lote1024=float(np.median(t)) * 1e3 / 1024)


def pendiente(x, y):
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


if __name__ == "__main__":
    import pandas as pd
    torch.manual_seed(0)
    torch.set_num_threads(4)
    filas = [dict(serie="longitud", **medir(L, (64, 64, 16))) for L in (168, 240, 336, 504, 672)]
    filas += [dict(serie="ancho", **medir(168, (c, c, c // 4))) for c in (16, 32, 64, 128)]
    d = pd.DataFrame(filas)
    d.to_csv("resultados/e3/complejidad_padre.csv", index=False)
    print(d.to_string(index=False))
    for serie, var in [("longitud", "L"), ("ancho", "ancho")]:
        s = d[d.serie == serie]
        print(f"{serie}: pendiente log-log FLOPs {pendiente(s[var], s.mflops_ventana):.2f}, "
              f"tiempo {pendiente(s[var], s.ms_ventana_lote1024):.2f}, parámetros {pendiente(s[var], s.parametros):.2f}")
