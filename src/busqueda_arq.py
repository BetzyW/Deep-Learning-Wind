"""Búsqueda de arquitectura del E2 (sección 14 del enunciado y sección 5 del preregistro, con la enmienda 1).

Para cada modelo sin arquitectura definida:
  1. Ajuste previo de hiperparámetros no arquitectónicos: T = 20 configuraciones aleatorias (semilla 7) con la
     arquitectura de referencia w = (64, 64, 64), b = (0, 1, 0).
  2. Fase 1 (cribado): las 1 000 arquitecturas (W³ × {0,1}³, W = {16, 32, 64, 128, 256}), una semilla (2000),
     máximo 12 épocas, paciencia 3.
Ambas etapas usan el último origen de validación (2024-10) y el 10 % de las ventanas de ajuste (enmienda 1).
Todo se guarda fila a fila en resultados/e2/busqueda/ y el script se reanuda donde quedó.
Uso: python src/busqueda_arq.py [modelos...]   (por defecto, todos en orden de costo)
"""
import csv, itertools, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from origen_movil import leer_config, objetivos, pliegues, preparar_panel  # noqa: E402
from modelos_e2 import Datos, entrenar  # noqa: E402

ORDEN = ["MLP", "CNN", "GRU", "LSTM", "TCN", "KAN", "Transformer", "Mamba"]
W = [16, 32, 64, 128, 256]
REF = dict(w=(64, 64, 64), b=(0, 1, 0))
FRAC = 0.10
CARPETA = "resultados/e2/busqueda"
# Enmienda 2 (8-nov-2026): por costo, las recurrentes evalúan en la fase 1 una muestra aleatoria simple de 200 de las
# 1 000 arquitecturas, fijada con la semilla 11 sin mirar resultados. Filas ya calculadas fuera de la muestra se
# conservan en el CSV pero no entran en la selección de finalistas.
MUESTRA = {"GRU": 200, "LSTM": 200}
SEMILLA_MUESTRA = 11


def muestra_arq(arquitecturas, nombre):
    if nombre not in MUESTRA:
        return arquitecturas
    idx = np.sort(np.random.default_rng(SEMILLA_MUESTRA).choice(len(arquitecturas), MUESTRA[nombre], replace=False))
    return [arquitecturas[i] for i in idx]


def ya_hechos(ruta, clave):
    if not os.path.exists(ruta):
        return {}
    with open(ruta, newline="", encoding="utf-8") as f:
        return {r[clave]: r for r in csv.DictReader(f)}


def anotar(ruta, fila):
    nuevo = not os.path.exists(ruta)
    with open(ruta, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fila))
        if nuevo:
            w.writeheader()
        w.writerow(fila)


def correr(nombre, datos, f, y_ev, semilla, hp_arq, hp, epocas):
    t0 = time.time()
    try:
        r = entrenar(nombre, datos, f, semilla, hp=hp_arq, lr=hp["lr"], wd=hp["wd"], bs=hp["bs"], epocas=epocas,
                     paciencia=3, frac=FRAC)
        rmse = float(np.sqrt(np.mean((r["pred"] - y_ev) ** 2)))
        ok = np.isfinite(rmse)
        return dict(rmse_val=rmse if ok else "", rmse_parada=r["rmse_parada"], rmse_train=r["historia_train"][-1],
                    epocas=r["epocas"], segundos=round(r["segundos"], 1), parametros=r["parametros"],
                    estado="ok" if ok else "excluida: NaN")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return dict(rmse_val="", rmse_parada="", rmse_train="", epocas=0, segundos=round(time.time() - t0, 1),
                    parametros="", estado="excluida: memoria GPU")


if __name__ == "__main__":
    modelos = sys.argv[1:] or ORDEN
    # esperar a que termine el piloto para no competir por la GPU
    while not (os.path.exists("resultados/e2/piloto.json")
               and "resumen" in json.load(open("resultados/e2/piloto.json"))):
        time.sleep(60)
    os.makedirs(CARPETA, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg = leer_config()
    p = preparar_panel(cfg)
    f = pliegues(p, cfg, "validacion")[-1]
    datos = Datos(p, f)
    y_ev = objetivos(p, f["evaluacion"])[:, 0]
    rng = np.random.default_rng(7)
    espacio_hp = [dict(id=i, lr=float(10 ** rng.uniform(-4, np.log10(3e-3))), wd=float(rng.choice([0, 1e-5, 1e-4])),
                       bs=int(rng.choice([512, 1024, 2048]))) for i in range(20)]
    arquitecturas = [dict(id=f"w{a}-{b}-{c}_d{x}{y}{z}", w=(a, b, c), b=(x, y, z))
                     for (a, b, c) in itertools.product(W, repeat=3) for (x, y, z) in itertools.product([0, 1], repeat=3)]
    for nombre in modelos:
        ruta_hp = f"{CARPETA}/{nombre}_hp.csv"
        hechos = ya_hechos(ruta_hp, "id")
        for hp in espacio_hp:
            if str(hp["id"]) in hechos:
                continue
            fila = dict(modelo=nombre, **hp, **correr(nombre, datos, f, y_ev, 2000, REF, hp, 20))
            anotar(ruta_hp, fila)
            print(f"[{time.strftime('%H:%M')}] {nombre} hp {hp['id']:2d} lr={hp['lr']:.1e} wd={hp['wd']:.0e} "
                  f"bs={hp['bs']} → {fila['rmse_val']} ({fila['segundos']} s)", flush=True)
        filas = [r for r in ya_hechos(ruta_hp, "id").values() if r["estado"] == "ok"]
        mejor = min(filas, key=lambda r: float(r["rmse_val"]))
        hp = dict(lr=float(mejor["lr"]), wd=float(mejor["wd"]), bs=int(mejor["bs"]))
        json.dump(hp, open(f"{CARPETA}/{nombre}_hp_elegido.json", "w"))
        ruta_f1 = f"{CARPETA}/{nombre}_fase1.csv"
        hechos = ya_hechos(ruta_f1, "id")
        elegidas = muestra_arq(arquitecturas, nombre)
        json.dump([a["id"] for a in elegidas], open(f"{CARPETA}/{nombre}_muestra_fase1.json", "w"))
        for a in elegidas:
            if a["id"] in hechos:
                continue
            fila = dict(modelo=nombre, id=a["id"], w1=a["w"][0], w2=a["w"][1], w3=a["w"][2], b1=a["b"][0],
                        b2=a["b"][1], b3=a["b"][2], **correr(nombre, datos, f, y_ev, 2000, dict(w=a["w"], b=a["b"]),
                                                              hp, 12))
            anotar(ruta_f1, fila)
            print(f"[{time.strftime('%H:%M')}] {nombre} fase 1 {a['id']} → {fila['rmse_val']} ({fila['segundos']} s)",
                  flush=True)
