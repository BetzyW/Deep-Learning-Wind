"""Chronos y Moirai sin entrenar (cero disparos) frente a su versión con cabezal y a los modelos del benchmark
(enmienda 9; análisis secundario, hecho después de abrir la prueba).

Toma los pronósticos de `preentrenados.py` (mediana de las trayectorias, cuantiles 0.1 y 0.9) en los 4 orígenes de
prueba de 2025 y los compara con las corridas finales sobre **las mismas ventanas** que usa `estadistica_e2.py`
(la intersección de todos los modelos), con la estación como unidad de independencia:
  * RMSE por origen y cobertura del intervalo del 80 % (cuantiles 0.1–0.9), global y con viento observado ≥ 6 m/s;
  * Wilcoxon de rangos con signo sobre la diferencia de RMSE por estación frente al cabezal, a Ridge, a la
    persistencia y al mejor modelo desde cero (Transformer en RQ2), y TOST con ±δ frente a este último.
Salida: resultados/e2/preentrenados/cero_disparos.json
Uso: python src/cero_disparos.py      (ml_env; no usa GPU)
"""
import glob, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from origen_movil import leer_config, preparar_panel  # noqa: E402
from estadistica_e2 import DELTA, cargar, tost, wilcoxon_estaciones  # noqa: E402

CARPETA = "resultados/e2/preentrenados"
RIVALES = ["Transformer", "TCN", "Ridge", "Persistencia"]


def cargar_cero(p, nombre):
    filas = []
    for a in sorted(glob.glob(f"{CARPETA}/{nombre}_prueba_*.npz")):
        z = np.load(a)
        filas.append(pd.DataFrame(dict(tau=z["tau"], pred=z["q"][:, 1], q10=z["q"][:, 0], q90=z["q"][:, 2],
                                       origen=os.path.basename(a)[-14:-4])))
    d = pd.concat(filas, ignore_index=True)
    d["estacion"] = p.estacion[d.tau]
    d["y"] = p.X[d.tau + p.horizontes[0], 0]
    return d


def rmse_est(d):
    return d.assign(e2=(d.pred - d.y) ** 2).groupby("estacion").e2.mean().pipe(np.sqrt)


if __name__ == "__main__":
    p = preparar_panel(leer_config())
    modelos = [os.path.basename(c) for c in sorted(glob.glob("resultados/e2/finales/*")) if os.path.isdir(c)]
    datos = {m: cargar(p, m) for m in modelos}
    for m in ["Chronos", "Moirai"]:
        datos[f"{m}_cero"] = cargar_cero(p, m)
    clave = None
    for d in datos.values():                       # mismas ventanas que estadistica_e2.py, más las de cero disparos
        i = d.dropna(subset=["y"]).set_index("tau").index
        clave = i if clave is None else clave.intersection(i)
    for m in datos:
        datos[m] = datos[m].set_index("tau").loc[clave].reset_index()
    origen = pd.Series(datos["Moirai_cero"].origen.to_numpy(), index=datos["Moirai_cero"].tau)
    r = dict(ventanas=int(len(clave)), delta=DELTA)
    est = {m: rmse_est(datos[m]) for m in ["Chronos", "Moirai", "Chronos_cero", "Moirai_cero"] + RIVALES}
    for m in ["Chronos", "Moirai"]:
        c = datos[f"{m}_cero"]
        fuerte = c.y >= 6
        dentro = (c.y >= c.q10) & (c.y <= c.q90)
        r[m] = dict(
            rmse_por_origen={o: float(np.sqrt(((g.pred - g.y) ** 2).mean())) for o, g in c.groupby("origen")},
            rmse_cabezal_por_origen={o: float(np.sqrt(((g.pred - g.y) ** 2).mean()))
                                     for o, g in datos[m].assign(origen=origen.loc[datos[m].tau].to_numpy())
                                     .groupby("origen")},
            rmse_medio_estaciones=float(est[f"{m}_cero"].mean()),
            cobertura_80=float(dentro.mean()), cobertura_80_fuerte=float(dentro[fuerte].mean()),
            sesgo_fuerte=float((c.pred - c.y)[fuerte].mean()), ventanas_fuertes=int(fuerte.sum()),
            contra={})
        for rival in [m] + RIVALES:                  # [m] = la misma arquitectura con cabezal
            w = wilcoxon_estaciones(est[f"{m}_cero"], est[rival])
            d = (est[f"{m}_cero"] - est[rival]).to_numpy()
            r[m]["contra"]["cabezal" if rival == m else rival] = dict(**w, tost=tost(d))
    for m in RIVALES:
        r[m] = dict(rmse_medio_estaciones=float(est[m].mean()),
                    sesgo_fuerte=float((datos[m].pred - datos[m].y)[datos[m].y >= 6].mean()),
                    rmse_por_origen={o: float(np.sqrt(((g.pred - g.y) ** 2).mean()))
                                     for o, g in datos[m].assign(origen=origen.loc[datos[m].tau].to_numpy())
                                     .groupby("origen")})
    json.dump(r, open(f"{CARPETA}/cero_disparos.json", "w"), indent=1, ensure_ascii=False)
    for m in ["Chronos", "Moirai"]:
        print(m, json.dumps({k: v for k, v in r[m].items() if k != "contra"}, ensure_ascii=False))
        for k, v in r[m]["contra"].items():
            print(f"   vs {k:12s} Δ medio {v['diferencia_media']:+.4f}  p Wilcoxon {v['p']:.2g}  "
                  f"p TOST {v['tost']['p']:.3g}")
    for m in RIVALES:
        print(m, r[m])
