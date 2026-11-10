"""Baselines del E2 (paso S5) en los orígenes de VALIDACIÓN (2024). La prueba (2025) no se toca hasta el preregistro.

Triviales (sin ajuste): persistencia (= naive estacional diaria, porque h = 24), naive estacional semanal
(v en τ+24−168) y media de la misma hora en los últimos 7 días.
Clásicos fuertes (reentrenados en cada origen con la ventana expansiva): ridge y gradient boosting
(HistGradientBoostingRegressor) sobre los rezagos de `origen_movil.rezagos`, un solo modelo global para todas las
estaciones. Pendiente: ARIMA/ETS por estación.
Uso: python src/baselines.py [validacion|prueba] → resultados/e2/baselines_{etapa}.csv
Con "prueba" (solo para las corridas finales) se guardan también las predicciones por ventana en
resultados/e2/finales/{baseline}/{origen}.npz, con el mismo formato que los modelos de Deep Learning.
"""
import os, sys, time
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from origen_movil import leer_config, objetivos, pliegues, preparar_panel, rezagos  # noqa: E402


def ets_por_estacion(p, origen, ev, semanas_ajuste=8, minimo=168):
    """ETS(A,N,A) con estacionalidad diaria, ajustado por estación con las últimas `semanas_ajuste` semanas antes del
    origen (statsmodels); luego el filtro corre hora a hora con esos parámetros fijos, así el pronóstico en τ solo usa
    datos hasta τ. Pronóstico a 24 h: nivel en τ + componente estacional de la misma hora. Si una hora falta, el
    estado no se actualiza. Estaciones sin datos suficientes antes del origen: se usa la persistencia."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    v = p.X[:, 0].astype(float)
    pred = v[ev].copy()                                              # respaldo: persistencia
    T = np.datetime64(origen, "h")
    inicio_est = {e: i for i, e in reversed(list(enumerate(p.estacion)))}
    n_ok = 0
    for e in np.unique(p.estacion[ev]):
        a = inicio_est[e]
        filas = a + np.flatnonzero(p.estacion[a:] == e)              # filas de la estación (contiguas)
        b = filas[-1] + 1
        previas = filas[p.tiempo[filas] < T]
        observadas = previas[~np.isnan(v[previas])]
        if len(observadas) < minimo:
            continue
        # tramo continuo más largo (tras rellenar huecos ≤ 6 h) en los últimos 6 meses antes del origen,
        # recortado a sus últimas `semanas_ajuste` semanas
        reciente = previas[-26 * 168:]
        yr = pd.Series(v[reciente]).interpolate(limit=6, limit_area="inside").to_numpy()
        ok = np.concatenate([[0], (~np.isnan(yr)).astype(int), [0]])
        bordes = np.flatnonzero(np.diff(ok))
        tramos = list(zip(bordes[::2], bordes[1::2]))
        if not tramos:
            continue
        i, j = max(tramos, key=lambda ab: ab[1] - ab[0])
        if j - i < minimo:
            continue
        i = max(i, j - semanas_ajuste * 168)
        antes, y = reciente[i:j], yr[i:j]
        fit = ExponentialSmoothing(y, trend=None, seasonal="add", seasonal_periods=24,
                                   initialization_method="estimated").fit()
        al, ga = fit.params["smoothing_level"], fit.params["smoothing_seasonal"]
        nivel = fit.level[-1]
        estac = list(fit.season[-24:])                               # estación[k] = componente de la hora t-23+k
        i0 = antes[-1] + 1
        ev_e = set(ev[p.estacion[ev] == e].tolist())
        salida = {}
        for t in range(i0, b):
            x = v[t]
            s_t = estac[0]                                           # componente estacional de la hora t
            if not np.isnan(x):
                nuevo = al * (x - s_t) + (1 - al) * nivel
                estac.append(ga * (x - nuevo) + (1 - ga) * s_t)
                nivel = nuevo
            else:
                estac.append(s_t)
            estac.pop(0)
            if t in ev_e:
                salida[t] = nivel + estac[-1]                        # misma hora, 24 h después
        m = np.isin(ev, list(salida))
        pred[m] = [salida[t] for t in ev[m]]
        n_ok += 1
    return pred, n_ok


def triviales(p, idx):
    v = p.X[:, 0]
    h = p.horizontes[0]
    misma_hora = [idx + h - 24 * k for k in range(1, 8)]                 # τ+h−24k, k = 1…7 (todas ≤ τ)
    return {"Persistencia": v[idx], "Naive semanal": v[idx + h - 168],
            "Media misma hora 7 d": np.mean([v[i] for i in misma_hora], 0)}


if __name__ == "__main__":
    etapa = sys.argv[1] if len(sys.argv) > 1 else "validacion"
    cfg = leer_config()
    p = preparar_panel(cfg)
    assert p.horizontes == [24], "este script asume un único horizonte de 24 h"
    filas, preds = [], []
    for f in pliegues(p, cfg, etapa):
        tr, ev = f["entrenamiento"], f["evaluacion"]
        y_tr, y_ev = objetivos(p, tr)[:, 0], objetivos(p, ev)[:, 0]
        X_tr, X_ev = rezagos(p, tr, cfg["rezagos_vel"]), rezagos(p, ev, cfg["rezagos_vel"])
        modelos = triviales(p, ev)
        te = time.time()
        modelos["ETS por estación"], n_ets = ets_por_estacion(p, f["origen"], ev)
        print(f"  ETS: {n_ets} estaciones ajustadas en {time.time() - te:.0f} s", flush=True)
        t0 = time.time()
        modelos["Ridge"] = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X_tr, y_tr).predict(X_ev)
        t1 = time.time()
        gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=63,
                                           early_stopping=False, random_state=0)
        modelos["Gradient boosting"] = gb.fit(X_tr, y_tr).predict(X_ev)
        print(f"origen {f['origen']}: entrenamiento {len(tr):,}, evaluación {len(ev):,} | ridge {t1 - t0:.0f} s, "
              f"boosting {time.time() - t1:.0f} s", flush=True)
        for m, yhat in modelos.items():
            if etapa == "prueba":
                carpeta = f"resultados/e2/finales/{m.replace(' ', '_')}"
                os.makedirs(carpeta, exist_ok=True)
                np.savez_compressed(f"{carpeta}/{f['origen']}.npz", tau=ev, pred=np.asarray(yhat, dtype=np.float32))
            e = yhat - y_ev
            filas.append(dict(origen=f["origen"], modelo=m, n=len(ev), sse=float((e ** 2).sum()),
                              sae=float(np.abs(e).sum())))
    t = pd.DataFrame(filas)
    t["rmse"], t["mae"] = np.sqrt(t.sse / t.n), t.sae / t.n
    # resumen juntando las ventanas de los 4 orígenes (no promedio simple de RMSE: los pliegues son desiguales)
    r = t.groupby("modelo")[["n", "sse", "sae"]].sum()
    r["rmse"], r["mae"] = np.sqrt(r.sse / r.n), r.sae / r.n
    r["mase_vs_persistencia"] = r.mae / r.loc["Persistencia", "mae"]
    r = r.sort_values("rmse")
    os.makedirs("resultados/e2", exist_ok=True)
    t.round(4).to_csv(f"resultados/e2/baselines_{etapa}_por_origen.csv", index=False)
    r.round(4).to_csv(f"resultados/e2/baselines_{etapa}.csv")
    print(t.pivot(index="modelo", columns="origen", values="rmse").round(3).to_string())
    print(r[["n", "rmse", "mae", "mase_vs_persistencia"]].round(3).to_string())
