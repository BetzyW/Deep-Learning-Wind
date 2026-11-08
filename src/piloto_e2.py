"""Piloto de potencia del E2 (sección 9.2): 5 semillas de DLinear, TCN y Transformer (referencia (64,64,64), (0,1,0))
en el último origen de validación (2024-10). Estima σ_d de las diferencias pareadas de RMSE entre modelos y el número
de ejecuciones R necesario para detectar δ = 0.02 m/s con potencia 0.90 (α = 0.05, bilateral).
Uso: python src/piloto_e2.py → resultados/e2/piloto.json
"""
import json, os, sys
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from origen_movil import leer_config, objetivos, pliegues, preparar_panel  # noqa: E402
from modelos_e2 import Datos, entrenar  # noqa: E402

DELTA, ALFA, POTENCIA = 0.02, 0.05, 0.90
SEMILLAS = range(3000, 3005)

if __name__ == "__main__":
    cfg = leer_config()
    p = preparar_panel(cfg)
    f = pliegues(p, cfg, "validacion")[-1]
    datos = Datos(p, f)
    y_ev = objetivos(p, f["evaluacion"])[:, 0]
    res = []
    os.makedirs("resultados/e2", exist_ok=True)
    for s in SEMILLAS:
        for nombre in ["DLinear", "TCN", "Transformer"]:
            r = entrenar(nombre, datos, f, s)
            r["rmse_val"] = float(np.sqrt(np.mean((r.pop("pred") - y_ev) ** 2)))
            print(f"{nombre:12s} semilla {s} | RMSE val {r['rmse_val']:.4f} | épocas {r['epocas']} | "
                  f"{r['segundos']:.0f} s | {r['parametros']:,} parámetros", flush=True)
            res.append(r)
            json.dump(dict(corridas=res), open("resultados/e2/piloto.json", "w"), indent=1)
    rm = {n: np.array([r["rmse_val"] for r in res if r["modelo"] == n]) for n in ["DLinear", "TCN", "Transformer"]}
    z = stats.norm.ppf(1 - ALFA / 2) + stats.norm.ppf(POTENCIA)
    pares = {}
    for a, b in [("DLinear", "TCN"), ("DLinear", "Transformer"), ("TCN", "Transformer")]:
        sd = float(np.std(rm[a] - rm[b], ddof=1))
        pares[f"{a}-{b}"] = dict(media=float(np.mean(rm[a] - rm[b])), sigma_d=sd,
                                 R_necesario=int(np.ceil((z * sd / DELTA) ** 2)))
    resumen = dict(rmse_medio={n: float(v.mean()) for n, v in rm.items()},
                   rmse_sd={n: float(v.std(ddof=1)) for n, v in rm.items()}, pares=pares,
                   R_final=max(6, max(v["R_necesario"] for v in pares.values())))
    json.dump(dict(corridas=res, resumen=resumen), open("resultados/e2/piloto.json", "w"), indent=1)
    print(json.dumps(resumen, indent=1))
