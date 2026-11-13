"""Inferencia estadística del E2 (sección 6 del preregistro) sobre las predicciones de prueba de 2025.

Entradas: resultados/e2/finales/{modelo}/{origen}_{semilla}.npz (Deep Learning) y {origen}.npz (baselines), con tau
(índice de la ventana en el panel) y pred. El pronóstico de un modelo de Deep Learning en cada ventana se resume por la
media de sus semillas declaradas (1000–1005); las semillas se usan además para la dispersión.

Pruebas:
  * Principal (familias F1 y F2): Wilcoxon de rangos con signo sobre la diferencia de RMSE por estación (la estación es
    la unidad de independencia), bilateral, con Holm dentro de cada familia; tamaño del efecto: diferencia mediana y
    correlación biserial de rangos.
  * Diebold–Mariano con la corrección de Harvey, Leybourne y Newbold (1997) sobre la serie horaria de diferencias de
    pérdida cuadrática (promedio entre estaciones en cada hora), varianza HAC de Newey–West con h − 1 = 23 rezagos.
  * Giacomini–White (2006) condicional con instrumentos (1, d_{t−1}): estadístico de Wald, χ² con 2 g.l.
  * Clark–West (2007) para cada modelo frente a la persistencia (modelo anidado), unilateral.
  * Model Confidence Set (Hansen, Lunde y Nason, 2011) sobre las pérdidas horarias, bootstrap por bloques (arch).
  * Friedman con post-hoc de Nemenyi (estaciones como bloques).
  * TOST con ±δ para la equivalencia (RQ2), sobre las diferencias de RMSE por estación.
Uso: python src/estadistica_e2.py → resultados/e2/estadistica/*.csv
"""
import glob, json, os, sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))

DELTA, ALFA, H = 0.02, 0.05, 24
SEM_E2 = range(1000, 1006)


def holm(p):
    p = np.asarray(p, float)
    orden = np.argsort(p)
    aj = np.empty_like(p)
    acum = 0.0
    for r, i in enumerate(orden):
        acum = max(acum, min(1.0, (len(p) - r) * p[i]))
        aj[i] = acum
    return aj


def hac_var(d, rezagos):
    """Varianza de largo plazo de Newey–West de la serie d (núcleo de Bartlett)."""
    d = d - d.mean()
    n = len(d)
    v = d @ d / n
    for k in range(1, rezagos + 1):
        v += 2 * (1 - k / (rezagos + 1)) * (d[k:] @ d[:-k]) / n
    return v


def diebold_mariano_hln(e1, e2, h=H):
    """H0: igual precisión (pérdida cuadrática). d = e1² − e2²; DM < 0 favorece al modelo 1."""
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    dm = d.mean() / np.sqrt(hac_var(d, h - 1) / n)
    k = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    t = k * dm
    return dict(estadistico=float(t), p=float(2 * stats.t.sf(abs(t), n - 1)), diferencia_media=float(d.mean()))


def giacomini_white(e1, e2):
    """Prueba condicional con instrumentos h_t = (1, d_t) para predecir d_{t+1} (Giacomini y White, 2006)."""
    d = e1 ** 2 - e2 ** 2
    Z = np.column_stack([np.ones(len(d) - 1), d[:-1]]) * d[1:, None]
    n = len(Z)
    zbar = Z.mean(0)
    omega = (Z - zbar).T @ (Z - zbar) / n
    W = n * zbar @ np.linalg.solve(omega, zbar)
    return dict(estadistico=float(W), p=float(stats.chi2.sf(W, Z.shape[1])))


def clark_west(e_bench, e_mod, f_bench, f_mod, h=H):
    """Clark y West (2007): el modelo grande (mod) contra el anidado (bench). H1: el grande es mejor (unilateral)."""
    f = e_bench ** 2 - (e_mod ** 2 - (f_bench - f_mod) ** 2)
    n = len(f)
    t = f.mean() / np.sqrt(hac_var(f, h - 1) / n)
    return dict(estadistico=float(t), p=float(stats.norm.sf(t)))


def wilcoxon_estaciones(r1, r2):
    """r1, r2: RMSE por estación (mismas estaciones). Diferencia = r1 − r2 (negativa favorece al modelo 1)."""
    d = np.asarray(r1) - np.asarray(r2)
    w = stats.wilcoxon(d, alternative="two-sided")
    rangos = stats.rankdata(np.abs(d))
    rbc = (rangos[d > 0].sum() - rangos[d < 0].sum()) / rangos.sum()
    return dict(estadistico=float(w.statistic), p=float(w.pvalue), diferencia_mediana=float(np.median(d)),
                diferencia_media=float(d.mean()), biserial=float(rbc), n_estaciones=len(d))


def tost(d, delta=DELTA):
    """Equivalencia de medias pareadas: H0 |μ_d| ≥ δ. Devuelve el mayor de los dos p unilaterales."""
    d = np.asarray(d)
    n, m, s = len(d), d.mean(), d.std(ddof=1)
    se = s / np.sqrt(n)
    p1 = stats.t.sf((m + delta) / se, n - 1)
    p2 = stats.t.cdf((m - delta) / se, n - 1)
    return dict(diferencia_media=float(m), p=float(max(p1, p2)), equivalentes=bool(max(p1, p2) < ALFA))


def rq2_equivalencia(rmse_est, preentrenados, desde_cero):
    """RQ2 (H2): el mejor preentrenado con ajuste fino frente al mejor entrenado desde cero, TOST con ±δ sobre la
    diferencia de RMSE por estación. 'Mejor' = menor RMSE medio entre estaciones (la misma unidad de la prueba)."""
    pre = min(preentrenados, key=lambda m: rmse_est[m].mean())
    cero = min(desde_cero, key=lambda m: rmse_est[m].mean())
    d = (rmse_est[pre] - rmse_est[cero]).to_numpy()
    se = d.std(ddof=1) / np.sqrt(len(d))
    t90 = stats.t.ppf(0.95, len(d) - 1)
    return dict(preentrenado=pre, desde_cero=cero, n_estaciones=len(d), **tost(d),
                ic90=[float(d.mean() - t90 * se), float(d.mean() + t90 * se)], delta=DELTA,
                wilcoxon=wilcoxon_estaciones(rmse_est[pre], rmse_est[cero]))


def familia_f3(datos, modelos, umbral=6.0, minimo=30, B=2000, semilla=0):
    """RQ3 (H3, familia F3): sesgo (predicho − observado) en las ventanas con observación ≥ umbral m/s. Por modelo:
    sesgo medio de cada estación con al menos `minimo` ventanas fuertes; Wilcoxon unilateral (H1: sesgo < 0, es decir,
    subestimación) entre estaciones, con Holm en la familia; sesgo agregado con IC 95 % por bootstrap de estaciones."""
    rng = np.random.default_rng(semilla)
    filas = []
    for m in modelos:
        d = datos[m]
        f = d[d.y >= umbral].assign(sesgo=lambda x: x.pred - x.y)
        g = f.groupby("estacion").sesgo.agg(["sum", "size"])
        g = g[g["size"] >= minimo]
        por_est = (g["sum"] / g["size"]).to_numpy()
        idx = rng.integers(0, len(g), (B, len(g)))
        boot = g["sum"].to_numpy()[idx].sum(1) / g["size"].to_numpy()[idx].sum(1)
        w = stats.wilcoxon(por_est, alternative="less")
        filas.append(dict(modelo=m, n_estaciones=len(g), n_ventanas=int(g["size"].sum()),
                          sesgo_agregado=float(g["sum"].sum() / g["size"].sum()),
                          ic95_inf=float(np.quantile(boot, .025)), ic95_sup=float(np.quantile(boot, .975)),
                          sesgo_mediano_estaciones=float(np.median(por_est)),
                          estaciones_subestiman=float((por_est < 0).mean()), wilcoxon_p=float(w.pvalue)))
    t = pd.DataFrame(filas)
    t["wilcoxon_p_holm"] = holm(t.wilcoxon_p)
    t["subestima"] = t.wilcoxon_p_holm < ALFA
    return t


def friedman_nemenyi(R):
    """R: DataFrame estaciones × modelos con el RMSE. Devuelve Friedman y la diferencia crítica de Nemenyi."""
    f = stats.friedmanchisquare(*[R[c].to_numpy() for c in R.columns])
    k, n = R.shape[1], R.shape[0]
    rangos = R.rank(axis=1).mean().sort_values()
    q = stats.studentized_range.ppf(1 - ALFA, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6 * n))
    return dict(estadistico=float(f.statistic), p=float(f.pvalue), rangos_medios=rangos.round(3).to_dict(),
                diferencia_critica=float(cd))


def mcs(perdidas, tam_bloque=168, reps=2000, alfa=0.10, semilla=0):
    from arch.bootstrap import MCS
    m = MCS(perdidas, size=alfa, reps=reps, block_size=tam_bloque, method="max", seed=semilla)
    m.compute()
    return dict(incluidos=list(m.included), excluidos=list(m.excluded),
                pvalores=m.pvalues["Pvalue"].round(4).to_dict())


# ---------------------------------------------------------------- carga de predicciones


def cargar(panel, nombre, carpeta="resultados/e2/finales"):
    """DataFrame con tau, estacion, tiempo, y, pred (media de semillas declaradas si es un modelo de Deep Learning)."""
    filas = []
    archivos = sorted(glob.glob(f"{carpeta}/{nombre}/*.npz"))
    por_origen = {}
    for a in archivos:
        base = os.path.basename(a)[:-4]
        origen, _, semilla = base.partition("_")
        if semilla and int(semilla) not in SEM_E2:
            continue
        z = np.load(a)
        por_origen.setdefault(origen, []).append((z["tau"], z["pred"]))
    for origen, lista in por_origen.items():
        tau = lista[0][0]
        pred = np.mean([p for _, p in lista], 0)
        filas.append(pd.DataFrame(dict(tau=tau, pred=pred, n_semillas=len(lista))))
    d = pd.concat(filas, ignore_index=True)
    d["estacion"] = panel.estacion[d.tau]
    d["tiempo"] = panel.tiempo[d.tau]
    d["y"] = panel.X[d.tau + panel.horizontes[0], 0]
    return d


if __name__ == "__main__":
    from origen_movil import leer_config, preparar_panel
    cfg = leer_config()
    p = preparar_panel(cfg)
    os.makedirs("resultados/e2/estadistica", exist_ok=True)
    modelos = [os.path.basename(c) for c in sorted(glob.glob("resultados/e2/finales/*")) if os.path.isdir(c)]
    datos = {m: cargar(p, m) for m in modelos}
    ref = datos["Persistencia"]
    clave = ref.set_index("tau").index
    for m in modelos:                                   # mismas ventanas para todos los modelos
        clave = clave.intersection(datos[m].set_index("tau").index)
    for m in modelos:
        datos[m] = datos[m].set_index("tau").loc[clave].reset_index()
    rmse_est = pd.DataFrame({m: datos[m].assign(e2=(datos[m].pred - datos[m].y) ** 2).groupby("estacion").e2.mean()
                             .pipe(np.sqrt) for m in modelos})
    rmse_est.to_csv("resultados/e2/estadistica/rmse_por_estacion.csv")
    # serie horaria de pérdidas (media entre estaciones en cada hora) para DM, GW, CW y MCS
    perd = pd.DataFrame({m: datos[m].assign(e2=(datos[m].pred - datos[m].y) ** 2).groupby("tiempo").e2.mean()
                         for m in modelos}).dropna()
    err = {m: datos[m].assign(e=datos[m].pred - datos[m].y).groupby("tiempo").e.mean().loc[perd.index].to_numpy()
           for m in modelos}
    top10 = [m for m in modelos if m not in {"Persistencia", "Naive_semanal", "Media_misma_hora_7_d",
                                              "ETS_por_estación", "Ridge", "Gradient_boosting"}]
    resultados = {}
    for fam, rival in [("F1", "Ridge"), ("F2", "Persistencia")]:
        filas = []
        for m in top10:
            w = wilcoxon_estaciones(rmse_est[m], rmse_est[rival])
            dm = diebold_mariano_hln(err[m], err[rival])
            gw = giacomini_white(err[m], err[rival])
            filas.append(dict(modelo=m, rival=rival, **{f"wilcoxon_{k}": v for k, v in w.items()},
                              dm_estadistico=dm["estadistico"], dm_p=dm["p"], gw_p=gw["p"]))
        t = pd.DataFrame(filas)
        t["wilcoxon_p_holm"] = holm(t.wilcoxon_p)
        t["supera_con_relevancia"] = (t.wilcoxon_p_holm < ALFA) & (t.wilcoxon_diferencia_media < -DELTA)
        t.to_csv(f"resultados/e2/estadistica/familia_{fam}.csv", index=False)
        resultados[fam] = t
    resultados["friedman"] = friedman_nemenyi(rmse_est[top10 + ["Ridge", "Persistencia"]])
    resultados["mcs"] = mcs(perd[top10 + ["Ridge", "Persistencia"]])
    # RQ2: equivalencia del mejor preentrenado con el mejor entrenado desde cero (TOST)
    pre = [m for m in ["Chronos", "Moirai"] if m in top10]
    resultados["rq2_tost"] = rq2_equivalencia(rmse_est, pre, [m for m in top10 if m not in pre])
    # RQ3 / F3: subestimación de los vientos fuertes (≥ 6 m/s), familia de los modelos del Top 10 con Holm;
    # Ridge y la persistencia se reportan como referencia, fuera de la familia
    f3 = familia_f3(datos, top10)
    ref = familia_f3(datos, ["Ridge", "Persistencia"]).assign(subestima=np.nan, wilcoxon_p_holm=np.nan)
    pd.concat([f3, ref]).to_csv("resultados/e2/estadistica/familia_F3.csv", index=False)
    otros = f3[(f3.modelo != "DLinear") & f3.subestima]
    resultados["rq3"] = dict(dlinear_subestima=bool(f3.set_index("modelo").subestima.get("DLinear", False)),
                             otros_que_subestiman=otros.modelo.tolist(),
                             h3_se_cumple=bool(f3.set_index("modelo").subestima.get("DLinear", False)
                                               and len(otros) >= 2))
    json.dump({k: v for k, v in resultados.items() if isinstance(v, dict)},
              open("resultados/e2/estadistica/resumen.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in resultados.items() if isinstance(v, dict)}, indent=1, ensure_ascii=False))
