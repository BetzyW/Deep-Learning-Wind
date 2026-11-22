"""Análisis de la ronda 2 de diseño (partición de desarrollo): sesgo con viento fuerte y dispersión entre anclas.

Para cada variante y origen de desarrollo:
* RMSE y sesgo medio (pred − y) en todas las ventanas y en las de viento observado ≥ 6 m/s;
* correlación de Spearman entre la dispersión entre anclas y el error absoluto (hipótesis del mecanismo);
* cobertura y ancho medio de intervalos conformales del 90 % calibrados con validación cruzada entre orígenes
  (se calibra con dos orígenes y se evalúa en el tercero): absolutos (|e| ≤ q) para todos y normalizados por la
  dispersión (|e| ≤ q·(s + ε)) para las variantes con K > 1.
Salida: resultados/e3/diseno_r2_analisis.csv y una tabla en pantalla.
Uso: python src/analisis_diseno_r2.py      (ml_env o dl_env; solo lee resultados/e3/diseno_r2/)
"""
import glob, os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

D = "resultados/e3/diseno_r2"
UMBRAL, ALFA = 6.0, 0.10


def cuantil(r):
    n = len(r)
    return np.quantile(r, min(1.0, np.ceil((n + 1) * (1 - ALFA)) / n))


archivos = sorted(glob.glob(os.path.join(D, "*.npz")))
datos = {}
for a in archivos:
    v, o = os.path.basename(a)[:-4].split("_", 1)
    z = np.load(a)
    datos.setdefault(v, {})[o] = (z["pred"], z["y"], z["disp"])

filas = []
for v, por_origen in datos.items():
    origenes = sorted(por_origen)
    for o in origenes:
        p, y, s = por_origen[o]
        e = p - y
        fuerte = y >= UMBRAL
        fila = dict(variante=v, origen=o, n=len(y), rmse=np.sqrt(np.mean(e ** 2)), sesgo=e.mean(),
                    n_fuerte=int(fuerte.sum()), rmse_fuerte=np.sqrt(np.mean(e[fuerte] ** 2)),
                    sesgo_fuerte=e[fuerte].mean())
        otros = [k for k in origenes if k != o]
        r_cal = np.concatenate([por_origen[k][0] - por_origen[k][1] for k in otros])
        q = cuantil(np.abs(r_cal))
        fila.update(cob_abs=np.mean(np.abs(e) <= q), ancho_abs=2 * q)
        if np.isfinite(s).all() and s.std() > 0:
            eps = np.concatenate([por_origen[k][2] for k in otros]).mean() * 0.1
            s_cal = np.concatenate([por_origen[k][2] for k in otros])
            qn = cuantil(np.abs(r_cal) / (s_cal + eps))
            fila.update(rho_disp_err=spearmanr(s, np.abs(e)).statistic,
                        cob_norm=np.mean(np.abs(e) <= qn * (s + eps)), ancho_norm=np.mean(2 * qn * (s + eps)),
                        cob_norm_fuerte=np.mean(np.abs(e[fuerte]) <= qn * (s[fuerte] + eps)),
                        cob_abs_fuerte=np.mean(np.abs(e[fuerte]) <= q))
        else:
            fila.update(cob_abs_fuerte=np.mean(np.abs(e[fuerte]) <= q))
        filas.append(fila)

t = pd.DataFrame(filas)
t.to_csv("resultados/e3/diseno_r2_analisis.csv", index=False)
res = t.drop(columns="origen").groupby("variante").mean(numeric_only=True)
pd.set_option("display.width", 200)
print(res.round(4).to_string())
