"""Frente de Pareto de la búsqueda y sesgo del ganador (sección 14.5–14.7 del enunciado).

1. Figura: RMSE de validación de la fase 1 frente al número de parámetros, para los 9 modelos con búsqueda. Se marcan
   el frente de Pareto (ninguna otra arquitectura tiene menos parámetros y menor RMSE), los finalistas y la elegida.
2. Sesgo del ganador: la "mejor" de la fase 2 se eligió por tener el menor RMSE en 2024-10-01. Si esa ventaja fuera
   real, se mantendría en los otros tres orígenes de validación (enmienda 8). Se compara la ventaja de la mejor sobre
   la elegida y sobre la peor en el origen de selección y en los otros tres.
Salida: figuras/e2_pareto.png y resultados/e2/busqueda_analisis/sesgo_ganador.csv
Uso: python src/pareto_ganador.py      (ml_env; no usa GPU)
"""
import contextlib, io, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from analisis_busqueda import pareto  # noqa: E402
from validacion_origenes import resumen, tres_finalistas  # noqa: E402

B = "resultados/e2/busqueda"
MODELOS = ["MLP", "CNN", "GRU", "LSTM", "TCN", "KAN", "Transformer", "Chronos", "Moirai"]


def figura():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ejes = plt.subplots(3, 3, figsize=(11, 9))
    for ax, m in zip(ejes.ravel(), MODELOS):
        d = pd.read_csv(f"{B}/{m}_fase1.csv")
        d = d[d.estado == "ok"]
        mu = f"{B}/{m}_muestra_fase1.json"
        if os.path.exists(mu):                         # solo la muestra de 200 entra en la elección (enmiendas 2 y 3)
            d = d[d.id.isin(json.load(open(mu)))]
        ids = pareto(d, "parametros")
        f2 = pd.read_csv(f"{B}/{m}_fase2.csv")
        cfg = json.load(open(f"resultados/e2/finales/{m}_configuracion.json"))["arquitectura"]["id"]
        ax.scatter(d.parametros, d.rmse_val, s=6, c="0.7", lw=0, label="fase 1")
        p = d.set_index("id").loc[ids].sort_values("parametros")
        ax.step(p.parametros, p.rmse_val, where="post", c="C0", lw=1.2, label="frente de Pareto")
        fin = d[d.id.isin(set(f2.id))]
        ax.scatter(fin.parametros, fin.rmse_val, s=18, facecolors="none", edgecolors="C1", label="finalistas")
        if cfg in set(d.id):
            e = d.set_index("id").loc[cfg]
            ax.scatter([e.parametros], [e.rmse_val], marker="*", s=120, c="C3", label="elegida")
        ax.set_xscale("log")
        ax.set_ylim(d.rmse_val.quantile(0.0) - 0.005, d.rmse_val.quantile(0.95))
        ax.set_title(f"{m} ({len(d)} arquitecturas)", fontsize=10)
        ax.tick_params(labelsize=8)
    for ax in ejes[-1]:
        ax.set_xlabel("parámetros (escala log)")
    for ax in ejes[:, 0]:
        ax.set_ylabel("RMSE de validación, fase 1")
    ejes[0, 0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig("figuras/e2_pareto.png", dpi=120)


def sesgo_ganador():
    with contextlib.redirect_stdout(io.StringIO()):
        t = resumen(MODELOS)
    filas = []
    for m, g in t.groupby("modelo", sort=False):
        g = g.set_index("papel")
        if "mejor" not in g.index:
            papel, r2 = tres_finalistas(m)
            mejor = "elegida"                          # la elegida es también la mejor de la fase 2
        else:
            mejor = "mejor"
        otros = ["2024-01-01", "2024-04-01", "2024-07-01"]
        for rival in [x for x in ["elegida", "peor"] if x in g.index and x != mejor]:
            filas.append(dict(modelo=m, rival=rival,
                              ventaja_origen_seleccion=float(g.loc[rival, "2024-10-01"] - g.loc[mejor, "2024-10-01"]),
                              ventaja_otros_origenes=float(g.loc[rival, otros].mean() - g.loc[mejor, otros].mean())))
    s = pd.DataFrame(filas)
    s.to_csv("resultados/e2/busqueda_analisis/sesgo_ganador.csv", index=False)
    print(s.round(4).to_string(index=False))
    print("ventaja media en el origen de selección", s.ventaja_origen_seleccion.mean().round(4),
          "| en los otros", s.ventaja_otros_origenes.mean().round(4),
          "| se mantiene en", int((s.ventaja_otros_origenes > 0).sum()), "de", len(s))
    return s


if __name__ == "__main__":
    figura()
    sesgo_ganador()
