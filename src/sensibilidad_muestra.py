"""Análisis de sensibilidad del muestreo de la fase 1 (enmienda 3, punto 4).

MLP y CNN evaluaron las 1 000 arquitecturas. Se les aplica la misma muestra de 200 (semilla 11) que usan GRU, LSTM,
TCN, KAN y Transformer y se compara: cuántos de los 10 finalistas de la fase 1 completa quedan dentro de los 10
finalistas de la muestra, la posición de la mejor arquitectura de la muestra en el ranking completo y la diferencia de
RMSE de validación entre la mejor de la muestra y la mejor de las 1 000.
Uso: python src/sensibilidad_muestra.py → resultados/e2/sensibilidad_muestra.json
"""
import itertools, json
import numpy as np
import pandas as pd

W, K = [16, 32, 64, 128, 256], 10
arq = [f"w{a}-{b}-{c}_d{x}{y}{z}" for (a, b, c) in itertools.product(W, repeat=3)
       for (x, y, z) in itertools.product([0, 1], repeat=3)]
muestra = {arq[i] for i in np.sort(np.random.default_rng(11).choice(len(arq), 200, replace=False))}
res = {}
for m in ["MLP", "CNN"]:
    d = pd.read_csv(f"resultados/e2/busqueda/{m}_fase1.csv")
    d = d[d.estado == "ok"].sort_values("rmse_val").reset_index(drop=True)
    d["rango"] = np.arange(1, len(d) + 1)
    top_completo = d.head(K)
    dm = d[d.id.isin(muestra)]
    top_muestra = dm.head(K)
    mejor_m = dm.iloc[0]
    res[m] = dict(arquitecturas_validas=len(d), en_muestra=len(dm),
                  finalistas_compartidos=int(len(set(top_completo.id) & set(top_muestra.id))),
                  mejor_completo=dict(id=d.id[0], rmse=float(d.rmse_val[0])),
                  mejor_muestra=dict(id=mejor_m.id, rmse=float(mejor_m.rmse_val), rango_en_las_1000=int(mejor_m.rango)),
                  diferencia_rmse=float(mejor_m.rmse_val - d.rmse_val[0]),
                  peor_finalista_muestra_rango=int(top_muestra.rango.max()),
                  de_rmse_entre_arquitecturas=float(d.rmse_val.std()))
    print(m, json.dumps(res[m], ensure_ascii=False))
json.dump(res, open("resultados/e2/sensibilidad_muestra.json", "w"), indent=1, ensure_ascii=False)
