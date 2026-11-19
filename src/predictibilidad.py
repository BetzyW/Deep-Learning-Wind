"""S2: predictibilidad intrínseca de cada estación, medida solo con el periodo de entrenamiento (antes de 2024).

Dos medidas complementarias que no dependen de ningún modelo:

* Entropía de permutación normalizada (Bandt y Pompe, 2002): frecuencia de los patrones de orden de m valores
  consecutivos separados `retardo` horas. Vale 1 si todos los patrones son igual de probables (ruido) y baja
  cuando la serie tiene estructura. Los empates (velocidades repetidas o ceros) sesgan la medida (Zunino et al.,
  2017), así que se rompen con un ruido uniforme menor que la resolución del sensor (0.01 m/s) y semilla fija.
* Predictibilidad espectral Ω (Goerg, 2013): 1 − entropía espectral normalizada del periodograma de Welch.
  Vale 0 para ruido blanco y se acerca a 1 cuando la potencia se concentra en pocas frecuencias (p. ej. el
  ciclo diario).

Se reporta además el RMSE de la persistencia a 24 h y su correlación de rangos con ambas medidas, también en el
periodo de entrenamiento, para ver si la predictibilidad intrínseca anticipa la dificultad de cada estación.
"""
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import signal, stats

sys.path.insert(0, os.path.dirname(__file__))
from origen_movil import leer_config  # noqa: E402

RESOLUCION = 0.01


def entropia_permutacion(x, m=5, retardo=1, rng=None):
    """Entropía de permutación normalizada en [0, 1]; ignora los patrones que contienen faltantes."""
    x = np.asarray(x, float)
    if rng is not None:
        x = x + rng.uniform(-RESOLUCION / 2, RESOLUCION / 2, len(x))
    span = (m - 1) * retardo + 1
    if len(x) < span:
        return np.nan, 0
    w = np.lib.stride_tricks.sliding_window_view(x, span)[:, ::retardo]
    w = w[~np.isnan(w).any(1)]
    if len(w) == 0:
        return np.nan, 0
    orden = np.argsort(w, axis=1, kind="stable")
    codigo = orden @ (m ** np.arange(m))                      # un entero distinto por cada patrón de orden
    _, cuentas = np.unique(codigo, return_counts=True)
    p = cuentas / cuentas.sum()
    return float(-(p * np.log(p)).sum() / math.log(math.factorial(m))), len(w)


def omega_espectral(x, nperseg=1024):
    """Ω de Goerg con el periodograma de Welch promediado sobre los tramos sin faltantes de longitud ≥ nperseg."""
    x = np.asarray(x, float)
    ok = ~np.isnan(x)
    cortes = np.flatnonzero(np.diff(np.concatenate([[0], ok.astype(int), [0]])))
    psd, peso = None, 0
    for a, b in zip(cortes[::2], cortes[1::2]):
        if b - a < nperseg:
            continue
        _, s = signal.welch(x[a:b] - x[a:b].mean(), nperseg=nperseg)
        psd = s * (b - a) if psd is None else psd + s * (b - a)
        peso += b - a
    if psd is None:
        return np.nan, 0
    p = psd[1:] / psd[1:].sum()                               # sin la frecuencia cero (la media)
    p = p[p > 0]
    return float(1 + (p * np.log(p)).sum() / math.log(len(psd) - 1)), peso


def calcular(cfg, hasta="2024-01-01", minimo_horas=2000, semilla=0):
    df = pd.read_csv(cfg["panel"], usecols=["CodigoEstacion", "hora", "vel"],
                     dtype={"CodigoEstacion": str}, parse_dates=["hora"])
    df = df[~df.CodigoEstacion.isin(cfg.get("excluir_estaciones", [])) & (df.hora < hasta)]
    rng = np.random.default_rng(semilla)
    filas = []
    for cod, g in df.groupby("CodigoEstacion", sort=True):
        v = g.set_index("hora").sort_index()
        v = v[~v.index.duplicated()].asfreq("h")["vel"].ffill(limit=cfg["ffill_max_horas"]).to_numpy()
        if np.isfinite(v).sum() < minimo_horas:
            continue
        fila = dict(estacion=cod, horas=int(np.isfinite(v).sum()), media=np.nanmean(v),
                    empates=float(np.mean(np.diff(v)[np.isfinite(np.diff(v))] == 0)))
        for m in (4, 5, 6):
            fila[f"PE_m{m}"], _ = entropia_permutacion(v, m, 1, rng)
        fila["PE_m5_r24"], _ = entropia_permutacion(v, 5, 24, rng)   # patrones entre días (misma hora)
        fila["PE_m5_sin_ruptura"], _ = entropia_permutacion(v, 5, 1, None)
        fila["omega"], _ = omega_espectral(v)
        e = v[24:] - v[:-24]
        fila["rmse_persistencia"] = float(np.sqrt(np.nanmean(e ** 2)))
        fila["rmse_relativo"] = fila["rmse_persistencia"] / np.nanstd(v)
        filas.append(fila)
    return pd.DataFrame(filas)


def referencias(n=20000, semilla=0):
    """Valores de control: ruido blanco (PE ≈ 1, Ω ≈ 0) y una senoide diaria con ruido (PE baja, Ω alta)."""
    rng = np.random.default_rng(semilla)
    t = np.arange(n)
    ruido = rng.normal(size=n)
    seno = np.sin(2 * np.pi * t / 24) + 0.3 * rng.normal(size=n)
    return pd.DataFrame([dict(serie=s, PE_m5=entropia_permutacion(x, 5)[0], omega=omega_espectral(x)[0])
                         for s, x in [("ruido blanco", ruido), ("seno diario + ruido", seno)]])


if __name__ == "__main__":
    cfg = leer_config(sys.argv[1] if len(sys.argv) > 1 else "configs/e2.json")
    print(referencias().round(3).to_string(index=False))
    t = calcular(cfg)
    os.makedirs("resultados/e2", exist_ok=True)
    t.to_csv("resultados/e2/predictibilidad.csv", index=False)
    print(f"\n{len(t)} estaciones con al menos 2000 horas antes de 2024")
    print(t.describe().loc[["mean", "std", "min", "50%", "max"]].round(3).to_string())
    print("\nCorrelación de Spearman con el RMSE relativo de la persistencia (RMSE / desviación de la serie):")
    for c in ["PE_m4", "PE_m5", "PE_m6", "PE_m5_r24", "omega"]:
        r = stats.spearmanr(t[c], t["rmse_relativo"], nan_policy="omit")
        print(f"  {c:10s} ρ = {r.statistic:+.3f}  p = {r.pvalue:.4f}")
    d = t["PE_m5"] - t["PE_m5_sin_ruptura"]
    print(f"\nEfecto de romper empates en PE_m5: media {d.mean():+.4f}, máximo {d.abs().max():.4f}")
