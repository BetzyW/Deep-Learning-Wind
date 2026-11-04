"""Ajuste de hiperparámetros: búsqueda aleatoria con el mismo presupuesto (N configuraciones) para cada modelo.
Cada configuración se entrena con una semilla fija y se evalúa en validación (2024). Prueba no se toca."""
import json, os, sys
import numpy as np
sys.path.insert(0, "src")
from datos import cargar, Lotes
from modelos import entrenar

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
ESPACIO = {
    "MLP": {"h1": [128, 256, 512], "h2": [64, 128, 256], "dropout": [0.0, 0.1, 0.3], "lr": [3e-4, 1e-3, 3e-3]},
    "LSTM": {"h": [32, 64, 128], "capas": [1, 2], "dropout": [0.0, 0.2], "lr": [3e-4, 1e-3, 3e-3]},
    "CNN": {"c1": [16, 32, 64], "c2": [32, 64, 128], "k": [3, 5, 7], "dropout": [0.0, 0.1, 0.3],
            "lr": [3e-4, 1e-3, 3e-3]},
}
d = cargar(); lotes = Lotes(d)
# si se interrumpió, se retoma: las configuraciones ya evaluadas no se repiten
res = json.load(open("resultados/ajuste.json")) if os.path.exists("resultados/ajuste.json") else []
for nombre, esp in ESPACIO.items():
    rng = np.random.default_rng(2026)
    vistas = set()
    while len([r for r in res if r["modelo"] == nombre]) < N:
        hp = {k: v[rng.integers(len(v))] for k, v in esp.items()}
        hp = {k: (v.item() if hasattr(v, "item") else v) for k, v in hp.items()}
        clave = tuple(sorted(hp.items()))
        if clave in vistas:
            continue
        vistas.add(clave)
        if any(r["modelo"] == nombre and r["hp"] == hp for r in res):
            continue
        r = entrenar(nombre, d, lotes, semilla=7, hp=hp, epocas=30, paciencia=3)
        print(f"{nombre:5s} {hp} | RMSE val {r['rmse_val']:.4f} | épocas {r['epocas']} | {r['segundos']:.0f} s", flush=True)
        res.append(dict(modelo=nombre, hp=hp, rmse_val=r["rmse_val"], epocas=r["epocas"], segundos=r["segundos"],
                        parametros=r["parametros"]))
        json.dump(res, open("resultados/ajuste.json", "w"), indent=1)
mejores = {m: min((r for r in res if r["modelo"] == m), key=lambda r: r["rmse_val"])["hp"] for m in ESPACIO}
json.dump(mejores, open("configs/hiperparametros.json", "w"), indent=2)
print("Mejores:", mejores)
