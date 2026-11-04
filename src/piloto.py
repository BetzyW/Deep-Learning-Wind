"""Piloto: 5 semillas por modelo con la configuración inicial, evaluadas solo en validación (2024)."""
import json, sys
sys.path.insert(0, "src")
from datos import cargar, Lotes
from modelos import entrenar

HP0 = {"MLP": {}, "LSTM": {}, "CNN": {}}
d = cargar(); lotes = Lotes(d)
res = []
for s in range(5):
    for nombre in HP0:
        r = entrenar(nombre, d, lotes, semilla=100 + s, hp=HP0[nombre])
        print(f"{nombre:5s} semilla {r['semilla']} | RMSE val {r['rmse_val']:.4f} | épocas {r['epocas']:2d} | "
              f"{r['segundos']:5.0f} s | parámetros {r['parametros']:,}", flush=True)
        res.append({k: v for k, v in r.items() if not k.startswith("pred")})
json.dump(res, open("resultados/piloto.json", "w"), indent=1)
