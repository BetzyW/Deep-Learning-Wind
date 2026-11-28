"""Estudios de robustez, eficiencia y calidad probabilística del cuadro 5 del enunciado del E3 en el panel principal,
con PADRE-TCN (V6) y la familia de contrastes, para el capítulo de evaluación ampliada.

1. Sensibilidad a hiperparámetros: RMSE de validación de las MISMAS 20 configuraciones (lr, wd, lote) que probó cada
   modelo en el ajuste previo (origen 2024-10): mínimo, mediana, máximo y rango.
2. Estabilidad por semilla: RMSE por semilla (media de los 4 orígenes de prueba), desviación, coeficiente de variación
   y prueba de Friedman entre semillas con los orígenes como bloques. Figura de cajas en book/figuras/e3_semillas.png.
3. Cambio de distribución y regímenes (pronóstico medio de las 10 semillas): RMSE por trimestre de 2025 y RMSE y sesgo
   por rango de la velocidad observada, para PADRE-TCN, la TCN, el Transformer y la persistencia.
4. Errores compartidos: correlación de los residuos de PADRE-TCN con los de la TCN y el Transformer, y fracción del 5 %
   de sus peores ventanas que también está entre el 5 % peor de cada uno.
5. Eficiencia y frontera de Pareto: modelos no dominados en (latencia por ventana en CPU, Winkler) y en
   (FLOPs por ventana, RMSE).
Salida: resultados/e3/ampliada.json y book/figuras/e3_semillas.png
Uso: python src/evaluacion_ampliada.py      (ml_env)
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))

FIN, CONF, E3 = "resultados/e2/finales", "resultados/e3/confirmatorias/V6", "resultados/e3"
ENTRENABLES = ["Transformer", "KAN", "TCN", "STGCN", "Chronos", "GRU", "Informer", "DLinear", "Moirai", "PatchTST"]
SEMILLAS = range(1000, 1010)
BINS = [0, 2, 4, 6, 99]


def corridas(m):
    d = pd.read_csv(f"{E3}/confirmatorias/V6.csv" if m == "PADRE" else f"{FIN}/{m}.csv")
    return d[d.semilla.isin(SEMILLAS)]


def pronostico_medio(carpeta, p, determinista=False):
    partes = []
    for o in ["2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01"]:
        if determinista:
            z = np.load(f"{carpeta}/{o}.npz"); tau, pred = z["tau"].astype(np.int64), z["pred"].astype(np.float64)
        else:
            zs = [np.load(f"{carpeta}/{o}_{s}.npz") for s in SEMILLAS]
            tau = zs[0]["tau"].astype(np.int64)
            pred = np.mean([z["pred"].astype(np.float64) for z in zs], 0)
        partes.append(pd.DataFrame(dict(tau=tau, pred=pred, origen=o)))
    d = pd.concat(partes, ignore_index=True)
    d["y"] = p.X[d.tau + p.horizontes[0], 0].astype(np.float64)
    return d.dropna(subset=["y"]).set_index("tau")


def no_dominados(df, a, b):
    return [m for m in df.index if not any((df.loc[o, a] <= df.loc[m, a]) and (df.loc[o, b] <= df.loc[m, b]) and
                                           ((df.loc[o, a] < df.loc[m, a]) or (df.loc[o, b] < df.loc[m, b]))
                                           for o in df.index if o != m)]


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    res = {}

    # 1) sensibilidad a hiperparámetros
    sens = {}
    for m in ["PADRE"] + ENTRENABLES:
        d = pd.read_csv(f"{E3}/PADRE_hp.csv" if m == "PADRE" else f"resultados/e2/busqueda/{m}_hp.csv")
        d = d[d.estado == "ok"] if "estado" in d else d
        r = d.rmse_val
        sens[m] = dict(configuraciones=int(len(r)), minimo=float(r.min()), mediana=float(r.median()),
                       maximo=float(r.max()), rango=float(r.max() - r.min()),
                       rho_lr=float(stats.spearmanr(d.lr, r).statistic))
    res["sensibilidad_hp"] = sens

    # 2) estabilidad por semilla
    est, cajas = {}, {}
    for m in ["PADRE"] + ENTRENABLES:
        d = corridas(m)
        por_sem = d.groupby("semilla").rmse.mean()
        tabla = d.pivot_table(index="origen", columns="semilla", values="rmse")
        est[m] = dict(media=float(por_sem.mean()), de=float(por_sem.std(ddof=1)),
                      cv=float(por_sem.std(ddof=1) / por_sem.mean()),
                      friedman_semillas_p=float(stats.friedmanchisquare(*[tabla[c] for c in tabla.columns]).pvalue))
        cajas[m] = por_sem.to_numpy()
    res["estabilidad_semillas"] = est
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    orden = sorted([m for m in cajas if m not in ("Chronos", "Moirai")], key=lambda m: np.median(cajas[m]))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bp = ax.boxplot([cajas[m] for m in orden], tick_labels=["PADRE-TCN" if m == "PADRE" else m for m in orden],
                    patch_artist=True, widths=0.6)
    for caja, m in zip(bp["boxes"], orden):
        caja.set_facecolor("#e69f00" if m == "PADRE" else "#cfe2f3")
    for i, m in enumerate(orden, 1):
        ax.scatter(np.full(len(cajas[m]), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(cajas[m])),
                   cajas[m], s=10, color="k", alpha=0.6, zorder=3)
    ax.axhspan(cajas["PADRE"].mean() - 0.02, cajas["PADRE"].mean() + 0.02, color="#e69f00", alpha=0.08,
               label="PADRE-TCN ± δ (0.02 m/s)")
    ax.set_ylabel("RMSE de prueba por semilla (m/s)\n(media de los 4 orígenes de 2025)")
    ax.set_title("Variación entre las 10 semillas (1000–1009), panel IDEAM")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig("book/figuras/e3_semillas.png", dpi=130)

    # 3) y 4) regímenes, trimestres y errores compartidos
    from origen_movil import leer_config, preparar_panel
    p = preparar_panel(leer_config())
    D = {"PADRE": pronostico_medio(CONF, p), "TCN": pronostico_medio(f"{FIN}/TCN", p),
         "Transformer": pronostico_medio(f"{FIN}/Transformer", p),
         "Persistencia": pronostico_medio(f"{FIN}/Persistencia", p, determinista=True)}
    comun = D["PADRE"].index
    for m in D:
        comun = comun.intersection(D[m].index)
    D = {m: d.loc[comun] for m, d in D.items()}
    y = D["PADRE"].y
    reg = {}
    for m, d in D.items():
        e = d.pred - d.y
        cortes = pd.cut(d.y, BINS, right=False, labels=["0–2", "2–4", "4–6", "≥ 6"])
        reg[m] = dict(por_rango={str(k): dict(n=int(len(g)), rmse=float(np.sqrt((g ** 2).mean())), sesgo=float(g.mean()))
                                 for k, g in e.groupby(cortes, observed=True)},
                      por_trimestre={o: float(np.sqrt((g ** 2).mean())) for o, g in e.groupby(d.origen)})
    res["regimenes"] = reg
    eP = D["PADRE"].pred - y
    peor = lambda e: set(e.abs().nlargest(int(0.05 * len(e))).index)
    res["errores_compartidos"] = {m: dict(correlacion=float(np.corrcoef(eP, D[m].pred - y)[0, 1]),
                                          solape_peor_5=float(len(peor(eP) & peor(D[m].pred - y)) / len(peor(eP))))
                                  for m in ["TCN", "Transformer", "Persistencia"]}

    # 5) Pareto
    c = pd.read_csv(f"{E3}/costo_cpu.csv").set_index("modelo")
    r = pd.read_csv(f"{E3}/estadistica/resumen_modelos.csv").set_index("modelo")
    df = pd.DataFrame({"latencia": c.ms_ventana_lote1024, "flops": c.mflops_ventana}).join(
        r[["rmse", "winkler"]], how="inner")
    df = df.drop(index=[m for m in ["Chronos", "Moirai"] if m in df.index])   # solo se mide su cabezal
    res["pareto"] = dict(modelos=list(df.index), latencia_winkler=no_dominados(df, "latencia", "winkler"),
                         flops_rmse=no_dominados(df, "flops", "rmse"))
    json.dump(res, open(f"{E3}/ampliada.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps({k: res[k] for k in ["errores_compartidos", "pareto"]}, indent=1, ensure_ascii=False))
    print(pd.DataFrame(sens).T.round(4).to_string())
    print(pd.DataFrame(est).T.round(4).to_string())
    for m in reg:
        print(m, {k: (v["n"], round(v["rmse"], 3), round(v["sesgo"], 3)) for k, v in reg[m]["por_rango"].items()},
              {k: round(v, 4) for k, v in reg[m]["por_trimestre"].items()})
