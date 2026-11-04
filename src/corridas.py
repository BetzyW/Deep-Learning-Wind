"""Corridas definitivas: R semillas por modelo con los hiperparámetros elegidos; se guardan las predicciones de 2025."""
import json, sys
import numpy as np
sys.path.insert(0, "src")
from datos import cargar, Lotes
from modelos import entrenar

R = int(sys.argv[1])
hp = json.load(open("configs/hiperparametros.json"))
# con un segundo argumento (configuración de datos) se corre el análisis de sensibilidad, solo en validación
sens = sys.argv[2] if len(sys.argv) > 2 else None
d = cargar(sens) if sens else cargar(); lotes = Lotes(d)
res, preds = [], {}
for s in range(R):
    for nombre in ["MLP", "LSTM", "CNN"]:
        r = entrenar(nombre, d, lotes, semilla=1000 + s, hp=hp[nombre], evaluar_prueba=not sens)
        if not sens:
            preds[f"{nombre}_{r['semilla']}"] = r["pred_prueba"].astype(np.float32)
        print(f"{nombre:5s} semilla {r['semilla']} | RMSE val {r['rmse_val']:.4f} | épocas {r['epocas']} | "
              f"{r['segundos']:.0f} s", flush=True)
        res.append({k: v for k, v in r.items() if not k.startswith("pred")})
        json.dump(res, open("resultados/sensibilidad_presion.json" if sens else "resultados/corridas.json", "w"), indent=1)
if not sens:
    np.savez_compressed("resultados/predicciones_prueba.npz", **preds)
