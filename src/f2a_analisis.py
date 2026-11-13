"""Ruta F2a del E2, parte sin GPU (ml_env): residuos, calibración, incertidumbre y estabilidad de los 3 mejores
modelos (TCN, Transformer, KAN; ver `f2a_gpu.py`), con Ridge como referencia clásica, más los IC por bootstrap de
estaciones del RMSE de todos los modelos (lista de verificación del Apéndice B).

Entradas: predicciones de prueba de `finales.py` (media de las semillas declaradas 1000–1005) y, para los intervalos,
las predicciones del último origen de validación de `f2a_gpu.py` (media de las mismas semillas).
* Residuos por estación (serie horaria de 2025): Ljung–Box, ARCH-LM y Jarque–Bera con 24 rezagos; BDS en las primeras
  2 000 horas (es O(n²)). Se reporta la proporción de estaciones que rechazan al 5 %.
* Mincer–Zarnowitz agregado (HAC con 23 rezagos) y proporción de estaciones que rechazan a = 0, b = 1.
* Intervalos conformes partidos por estación: |e| del cuarto trimestre de 2024 de cada estación → cuantiles 50, 80 y
  90 %. La dependencia temporal rompe la intercambiabilidad: se declara como limitación y se mide la cobertura por
  trimestre, por estación y por tercil de la predicción (cobertura condicional, sección 12.2).
* CRPS de la distribución predictiva ŷ + e (e: residuos de validación de la estación) y PIT con prueba KS.
* Giacomini–Rossi (ventanas de 30 días) de cada modelo frente a Ridge sobre la pérdida cuadrática horaria media.
Salida: resultados/e2/f2a/*.csv y analisis.json
Uso: python src/f2a_analisis.py
"""
import glob, json, os, sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from estadistica_e2 import SEM_E2, cargar, hac_var  # noqa: E402
from ruta_f2a import NIVELES, mincer_zarnowitz, pinball, valor_critico_gr  # noqa: E402

TOP3 = ["TCN", "Transformer", "KAN"]
REF = "Ridge"
SALIDA = "resultados/e2/f2a"


def residuos(e, rezagos=24):
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    from statsmodels.tsa.stattools import bds
    e = e[np.isfinite(e)]
    return dict(ljung_box=float(acorr_ljungbox(e, lags=[rezagos], return_df=True)["lb_pvalue"].iloc[0]),
                arch_lm=float(het_arch(e, nlags=rezagos)[1]), jarque_bera=float(stats.jarque_bera(e).pvalue),
                bds=float(np.atleast_1d(bds(e[:2000], max_dim=2)[1])[0]))


def crps_empirico(y, yhat, r, trozo=20000):
    """CRPS de F = ŷ + r (r: muestra de residuos) por la fórmula de energía: E|X − y| − ½ E|X − X'|."""
    r = np.asarray(r, float)
    medio = 0.5 * np.abs(r[:, None] - r[None]).mean()
    tot = 0.0
    for i in range(0, len(y), trozo):
        tot += np.abs(yhat[i:i + trozo, None] + r[None] - y[i:i + trozo, None]).mean(1).sum()
    return tot / len(y) - medio


def giacomini_rossi_d(d, ventana=24 * 30, h=24):
    d = np.asarray(d, float)
    n = len(d)
    s = np.sqrt(hac_var(d, h - 1))
    serie = np.array([d[i:i + ventana].mean() * np.sqrt(ventana) / s for i in range(0, n - ventana + 1, 24)])
    mu = ventana / n
    critico = valor_critico_gr(mu)
    return dict(maximo=float(np.abs(serie).max()), critico=critico, mu=float(mu),
                inestable=bool(np.abs(serie).max() > critico)), serie


def cargar_validacion(panel, nombre):
    archivos = [a for a in sorted(glob.glob(f"{SALIDA}/{nombre}/validacion_*_*.npz"))
                if int(os.path.basename(a)[:-4].rsplit("_", 1)[1]) in SEM_E2]
    if len(archivos) < len(SEM_E2):
        return None
    zs = [np.load(a) for a in archivos]
    d = pd.DataFrame(dict(tau=zs[0]["tau"], pred=np.mean([z["pred"] for z in zs], 0)))
    d["estacion"] = panel.estacion[d.tau]
    d["y"] = panel.X[d.tau + panel.horizontes[0], 0]
    return d.dropna(subset=["y"])


def bootstrap_rmse(d, B=2000, semilla=0):
    """IC 95 % percentil del RMSE con remuestreo de estaciones (la unidad de independencia)."""
    g = d.assign(e2=(d.pred - d.y) ** 2).groupby("estacion").e2.agg(["sum", "size"])
    rng = np.random.default_rng(semilla)
    idx = rng.integers(0, len(g), (B, len(g)))
    s, n = g["sum"].to_numpy()[idx].sum(1), g["size"].to_numpy()[idx].sum(1)
    v = np.sqrt(s / n)
    return float(np.sqrt(g["sum"].sum() / g["size"].sum())), float(np.quantile(v, .025)), float(np.quantile(v, .975))


if __name__ == "__main__":
    from origen_movil import leer_config, preparar_panel
    cfg = leer_config()
    p = preparar_panel(cfg)
    os.makedirs(SALIDA, exist_ok=True)
    res = {}

    # IC por bootstrap de estaciones para todos los modelos
    modelos = [os.path.basename(c) for c in sorted(glob.glob("resultados/e2/finales/*")) if os.path.isdir(c)]
    datos = {m: cargar(p, m).dropna(subset=["y"]) for m in modelos}
    ic = pd.DataFrame([dict(modelo=m, **dict(zip(["rmse", "ic_inf", "ic_sup"], bootstrap_rmse(datos[m]))))
                       for m in modelos]).sort_values("rmse")
    ic.to_csv(f"{SALIDA}/rmse_ic_bootstrap.csv", index=False)
    print(ic.round(4).to_string(index=False), flush=True)

    # residuos y Mincer–Zarnowitz por estación
    filas, mz = [], {}
    for m in TOP3 + [REF]:
        d = datos[m].sort_values(["estacion", "tiempo"])
        for est, g in d.groupby("estacion"):
            if len(g) < 500:
                continue
            filas.append(dict(modelo=m, estacion=est, n=len(g), **residuos((g.y - g.pred).to_numpy()),
                              mz_p=mincer_zarnowitz(g.y.to_numpy(), g.pred.to_numpy())["p_conjunto"]))
        mz[m] = mincer_zarnowitz(d.y.to_numpy(), d.pred.to_numpy())
        print(f"{m}: residuos y MZ listos", flush=True)
    r = pd.DataFrame(filas)
    r.to_csv(f"{SALIDA}/residuos_por_estacion.csv", index=False)
    res["residuos_proporcion_rechaza_5"] = (r.groupby("modelo")[["ljung_box", "arch_lm", "jarque_bera", "bds", "mz_p"]]
                                            .agg(lambda x: float((x < 0.05).mean())).to_dict(orient="index"))
    res["mincer_zarnowitz_agregado"] = mz

    # estabilidad: Giacomini–Rossi frente a Ridge sobre la pérdida horaria media entre estaciones
    perd = pd.DataFrame({m: datos[m].assign(e2=(datos[m].pred - datos[m].y) ** 2).groupby("tiempo").e2.mean()
                         for m in TOP3 + [REF]}).dropna()
    res["giacomini_rossi_vs_ridge"], series = {}, {}
    for m in TOP3:
        res["giacomini_rossi_vs_ridge"][m], series[m] = giacomini_rossi_d((perd[m] - perd[REF]).to_numpy())
    pd.DataFrame(series).to_csv(f"{SALIDA}/giacomini_rossi_series.csv", index=False)

    # incertidumbre: intervalos conformes por estación con residuos de validación
    res["intervalos"], cond = {}, []
    for m in TOP3 + [REF]:
        if m == REF:
            v = cargar(p, REF, carpeta="resultados/e2/finales_validacion") if os.path.isdir(
                "resultados/e2/finales_validacion") else None
        else:
            v = cargar_validacion(p, m)
        if v is None:
            print(f"{m}: faltan las predicciones de validación (f2a_gpu.py); intervalos pendientes", flush=True)
            continue
        rv = {est: (g.y - g.pred).to_numpy() for est, g in v.groupby("estacion")}
        d = datos[m][datos[m].estacion.isin(rv)].copy()
        d["e"] = d.y - d.pred
        filas, crps_tot, pit = [], 0.0, []
        for est, g in d.groupby("estacion"):
            e_val = np.abs(rv[est])
            n = len(e_val)
            q = {a: float(np.sort(e_val)[min(n - 1, int(np.ceil((n + 1) * a)) - 1)]) for a in NIVELES}
            for a, qa in q.items():
                dentro = (g.e.abs() <= qa).to_numpy()
                tercil = pd.qcut(g.pred, 3, labels=["bajo", "medio", "alto"]).astype(str).to_numpy()
                for etiqueta, mascara in [("todas", np.ones(len(g), bool))] + [
                        (f"tercil_{t}", tercil == t) for t in ["bajo", "medio", "alto"]]:
                    cond.append(dict(modelo=m, estacion=est, nivel=a, grupo=etiqueta, n=int(mascara.sum()),
                                     cobertura=float(dentro[mascara].mean()), ancho=2 * qa))
                lo, hi = (g.pred - qa).to_numpy(), (g.pred + qa).to_numpy()
                filas.append(dict(nivel=a, n=len(g), pin_inf=pinball(g.y.to_numpy(), lo, (1 - a) / 2) * len(g),
                                  pin_sup=pinball(g.y.to_numpy(), hi, (1 + a) / 2) * len(g)))
            rs = np.random.default_rng(0).choice(rv[est], min(300, len(rv[est])), replace=False)
            crps_tot += crps_empirico(g.y.to_numpy(), g.pred.to_numpy(), rs) * len(g)
            pit.append(np.searchsorted(np.sort(rv[est]), g.e.to_numpy()) / len(rv[est]))
        f = pd.DataFrame(filas).groupby("nivel").sum()
        c = pd.DataFrame([x for x in cond if x["modelo"] == m and x["grupo"] == "todas"])
        cob = c.assign(k=c.cobertura * c.n).groupby("nivel")[["k", "n"]].sum()
        anch = c.assign(k=c.ancho * c.n).groupby("nivel")[["k", "n"]].sum()
        pit = np.concatenate(pit)
        res["intervalos"][m] = dict(
            cobertura={str(a): float(cob.k[a] / cob.n[a]) for a in NIVELES},
            ancho_medio={str(a): float(anch.k[a] / anch.n[a]) for a in NIVELES},
            pinball_medio={str(a): float((f.pin_inf[a] + f.pin_sup[a]) / (2 * f.n[a])) for a in NIVELES},
            crps=float(crps_tot / len(d)), pit_ks_p=float(stats.kstest(pit, "uniform").pvalue),
            pit_histograma=np.histogram(pit, bins=10, range=(0, 1))[0].tolist())
        print(f"{m}: intervalos {res['intervalos'][m]['cobertura']}", flush=True)
    if cond:
        pd.DataFrame(cond).to_csv(f"{SALIDA}/cobertura_condicional.csv", index=False)
    json.dump(res, open(f"{SALIDA}/analisis.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False)[:4000])
