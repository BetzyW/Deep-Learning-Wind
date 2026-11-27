"""Adecuación del modelo original (condición 5 de la sección 8.4 del enunciado; Tablas A y B del E2): ajuste y
sobreajuste de PADRE-TCN (V6) en la prueba de 2025, con la TCN del E2 como referencia, medidos igual que en la ruta
F2a del E2 (`f2a_analisis.py`) pero con las 10 semillas.

* Brecha entre entrenamiento y parada temprana, por corrida (40 = 10 semillas × 4 orígenes): absoluta
  (RMSE de parada − RMSE de entrenamiento, m/s; la regla del protocolo del E3 exige |brecha| < δ = 0.02) y relativa
  (gap_ratio = |entrenamiento − parada| / parada, el criterio del E2, umbral 0.15).
* Consistencia entre validación y prueba: RMSE de parada medio frente a RMSE de prueba medio.
* Residuos del pronóstico medio de las 10 semillas, por estación (serie horaria de 2025, estaciones con ≥ 500 h):
  Ljung–Box(24), ARCH-LM(24), Jarque–Bera y BDS (primeras 2 000 h); proporción de estaciones que rechazan al 5 %.
* Mincer–Zarnowitz agregado (HAC, 23 rezagos): y = a + b ŷ, prueba conjunta a = 0, b = 1; y proporción de estaciones
  que rechazan.
Salida: resultados/e3/adecuacion.json
Uso: python src/adecuacion_padre.py      (ml_env)
"""
import glob, json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from f2a_analisis import residuos  # noqa: E402
from ruta_f2a import mincer_zarnowitz  # noqa: E402

MODELOS = {"PADRE": ("resultados/e3/confirmatorias/V6", "resultados/e3/confirmatorias/V6.csv"),
           "TCN": ("resultados/e2/finales/TCN", "resultados/e2/finales/TCN.csv")}
SEMILLAS = range(1000, 1010)


def pronostico_medio(carpeta, p):
    """Media de las 10 semillas por ventana, en los 4 orígenes de prueba."""
    partes = []
    for o in sorted({os.path.basename(a).split("_")[0] for a in glob.glob(f"{carpeta}/*_*.npz")}):
        zs = [np.load(f"{carpeta}/{o}_{s}.npz") for s in SEMILLAS]
        tau = zs[0]["tau"].astype(np.int64)
        assert all(np.array_equal(z["tau"].astype(np.int64), tau) for z in zs)
        pred = np.mean([z["pred"].astype(np.float64) for z in zs], 0)
        partes.append(pd.DataFrame(dict(tau=tau, pred=pred)))
    d = pd.concat(partes, ignore_index=True)
    h = p.horizontes[0]
    d["y"] = p.X[d.tau + h, 0].astype(np.float64)
    d["estacion"], d["tiempo"] = p.estacion[d.tau], p.tiempo[d.tau]
    return d.dropna(subset=["y"])


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    from origen_movil import leer_config, preparar_panel
    p = preparar_panel(leer_config())
    salida = {}
    for m, (carpeta, csv) in MODELOS.items():
        c = pd.read_csv(csv)
        c = c[c.semilla.isin(SEMILLAS)]
        brecha = c.rmse_parada - c.rmse_train
        gap = (c.rmse_train - c.rmse_parada).abs() / c.rmse_parada
        d = pronostico_medio(carpeta, p).sort_values(["estacion", "tiempo"])
        filas = []
        for est, g in d.groupby("estacion"):
            if len(g) < 500:
                continue
            filas.append(dict(**residuos((g.y - g.pred).to_numpy()),
                              mz=mincer_zarnowitz(g.y.to_numpy(), g.pred.to_numpy())["p_conjunto"]))
        r = pd.DataFrame(filas)
        salida[m] = dict(
            corridas=int(len(c)),
            brecha_absoluta_media=float(brecha.mean()), brecha_absoluta_max=float(brecha.abs().max()),
            gap_ratio_mediana=float(gap.median()), gap_ratio_max=float(gap.max()),
            rmse_entrenamiento_medio=float(c.rmse_train.mean()), rmse_parada_medio=float(c.rmse_parada.mean()),
            rmse_prueba_medio=float(c.rmse.mean()),
            estaciones=int(len(r)),
            proporcion_rechaza_5=r.agg(lambda x: float((x < 0.05).mean())).to_dict(),
            mincer_zarnowitz=mincer_zarnowitz(d.y.to_numpy(), d.pred.to_numpy()),
        )
        print(m, json.dumps(salida[m], indent=1), flush=True)
    json.dump(salida, open("resultados/e3/adecuacion.json", "w", encoding="utf-8"), indent=1)
