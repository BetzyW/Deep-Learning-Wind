"""Pruebas de las Tablas B y C del enunciado que faltaban en la ficha de ruta (F2a, series de tiempo).

Todas se calculan con resultados ya guardados; no se entrena nada.
* Tabla B, brecha entrenamiento–validación con margen: en cada corrida final (13 modelos × 4 orígenes × 6 semillas),
  g = |RMSE de entrenamiento − RMSE de parada| / RMSE de parada, el gap_ratio del curso, con δg = 0.15, el umbral
  del curso fijado antes de este análisis. Wilcoxon unilateral H0: mediana(g) ≥ δg.
* Tabla B, sensibilidad a la semilla: coeficiente de variación del RMSE de prueba entre las 6 semillas en cada origen,
  y Friedman entre semillas con los orígenes como bloques (H0: ninguna semilla es sistemáticamente distinta).
* Tabla B, validación frente a prueba: para los tres mejores (TCN, Transformer, KAN), RMSE por estación en el último
  origen de validación (2024-10-01) menos el de prueba (2025), con IC 95 % por bootstrap de estaciones; la misma
  diferencia en la persistencia, que no se ajusta, separa el cambio entre periodos del sobreajuste.
* Tabla A, Mincer–Zarnowitz agregado con errores HAC para los 13 modelos y los clásicos.
* Tabla C, Friedman con la corrección de Iman–Davenport, y Benjamini–Hochberg como sensibilidad de Holm en F1 y F2.
Salida: resultados/e2/estadistica/ruta_evidencias.json
Uso: python src/ruta_evidencias.py      (ml_env)
"""
import glob, json, os, sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from estadistica_e2 import SEM_E2  # noqa: E402

E = "resultados/e2"
TOP13 = ["Transformer", "KAN", "TCN", "STGCN", "Chronos", "GRU", "Informer", "DLinear", "Moirai", "PatchTST",
         "MLP", "CNN", "LSTM"]
DG = 0.15


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    r = np.empty_like(q); r[o] = np.minimum(q, 1)
    return r


def rmse_estacion(p, tau, pred):
    y = p.X[tau + p.horizontes[0], 0]
    d = pd.DataFrame(dict(est=p.estacion[tau], e2=(pred - y) ** 2)).dropna()
    return d.groupby("est").e2.mean().pipe(np.sqrt)


if __name__ == "__main__":
    r = {}
    filas = []
    for m in TOP13:
        d = pd.read_csv(f"{E}/finales/{m}.csv")
        d = d[d.semilla.isin(SEM_E2)]
        g = (d.rmse_train - d.rmse_parada).abs() / d.rmse_parada
        w = stats.wilcoxon(g - DG, alternative="less")
        cv = d.groupby("origen").rmse.agg(lambda x: x.std(ddof=1) / x.mean())
        kw = stats.friedmanchisquare(*[d[d.semilla == s].sort_values("origen").rmse for s in SEM_E2])
        filas.append(dict(modelo=m, corridas=len(d), gap_mediano=float(g.median()), gap_max=float(g.max()),
                          p_wilcoxon_gap=float(w.pvalue), cv_semillas_medio=float(cv.mean()),
                          cv_semillas_max=float(cv.max()), p_semillas=float(kw.pvalue),
                          rmse_media=float(d.groupby("semilla").rmse.mean().mean()),
                          rmse_de_semillas=float(d.groupby("semilla").rmse.mean().std(ddof=1))))
    t = pd.DataFrame(filas)
    from estadistica_e2 import holm
    t["p_wilcoxon_gap_holm"] = holm(t.p_wilcoxon_gap)
    r["brecha_y_semillas"] = t.to_dict("records")

    # validación frente a prueba, tres mejores
    from origen_movil import leer_config, preparar_panel
    from estadistica_e2 import cargar
    p = preparar_panel(leer_config())
    vt = {}
    rng = np.random.default_rng(0)
    for m in ["TCN", "Transformer", "KAN"]:
        zs = [np.load(a) for a in sorted(glob.glob(f"{E}/f2a/{m}/validacion_2024-10-01_*.npz"))]
        rv = rmse_estacion(p, zs[0]["tau"], np.mean([z["pred"] for z in zs], 0))
        dt = cargar(p, m)
        rt = rmse_estacion(p, dt.tau.to_numpy(), dt.pred.to_numpy())
        comun = rv.index.intersection(rt.index)
        dif = (rv.loc[comun] - rt.loc[comun]).to_numpy()
        boot = [rng.choice(dif, len(dif)).mean() for _ in range(10000)]
        vt[m] = dict(estaciones=int(len(comun)), diferencia_media=float(dif.mean()),
                     ic95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                     p_wilcoxon=float(stats.wilcoxon(dif).pvalue))
    # referencia: la misma diferencia para la persistencia (no se ajusta, así que solo puede reflejar el cambio
    # entre periodos, no sobreajuste a la validación)
    tau_v = zs[0]["tau"]
    rv = rmse_estacion(p, tau_v, p.X[tau_v, 0])
    rt = rmse_estacion(p, dt.tau.to_numpy(), p.X[dt.tau.to_numpy(), 0])
    comun = rv.index.intersection(rt.index)
    dif = (rv.loc[comun] - rt.loc[comun]).to_numpy()
    boot = [rng.choice(dif, len(dif)).mean() for _ in range(10000)]
    vt["Persistencia"] = dict(estaciones=int(len(comun)), diferencia_media=float(dif.mean()),
                              ic95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                              p_wilcoxon=float(stats.wilcoxon(dif).pvalue))
    r["validacion_vs_prueba"] = vt

    # Mincer–Zarnowitz agregado (HAC) para todos los modelos, para la columna de calibración de la tabla A.2
    from f2a_analisis import mincer_zarnowitz
    r["mincer_zarnowitz"] = {}
    for m in TOP13 + ["Ridge", "Gradient_boosting", "Persistencia"]:
        d = cargar(p, m).dropna(subset=["y"]).sort_values(["estacion", "tiempo"])
        r["mincer_zarnowitz"][m] = mincer_zarnowitz(d.y.to_numpy(), d.pred.to_numpy())
        print(m, r["mincer_zarnowitz"][m], flush=True)

    # Friedman con Iman–Davenport y BH frente a Holm
    res = json.load(open(f"{E}/estadistica/resumen.json"))
    k = len(res["friedman"]["rangos_medios"])
    N = 62
    chi2 = res["friedman"]["estadistico"]
    F = (N - 1) * chi2 / (N * (k - 1) - chi2)
    r["iman_davenport"] = dict(k=k, n_estaciones=N, F=float(F), gl=[k - 1, (k - 1) * (N - 1)],
                               p=float(stats.f.sf(F, k - 1, (k - 1) * (N - 1))))
    for fam in ("F1", "F2"):
        f = pd.read_csv(f"{E}/estadistica/familia_{fam}.csv")
        f["p_bh"] = bh(f.wilcoxon_p)
        r[f"bh_{fam}"] = dict(significativos_holm=int((f.wilcoxon_p_holm < 0.05).sum()),
                              significativos_bh=int((f.p_bh < 0.05).sum()),
                              cambian=f.loc[(f.wilcoxon_p_holm < 0.05) != (f.p_bh < 0.05), "modelo"].tolist())
    json.dump(r, open(f"{E}/estadistica/ruta_evidencias.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(t.round(4).to_string(index=False))
    for k_, v in r.items():
        if k_ != "brecha_y_semillas":
            print(k_, v)
