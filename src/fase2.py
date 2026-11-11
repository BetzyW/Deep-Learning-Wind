"""Fase 2 de la búsqueda de arquitectura (sección 14.4, paso A4): confirmación de los finalistas.

Para cada modelo:
  * Finalistas: las k mejores arquitecturas de la fase 1 por RMSE de validación, solo entre las que pertenecen a la
    muestra declarada (`{modelo}_muestra_fase1.json`) y terminaron bien, más la referencia (64, 64, 64) / (0, 1, 0).
  * Cada finalista se entrena con presupuesto completo (todas las ventanas de ajuste, máximo 50 épocas, paciencia 5),
    con las semillas 2001–2005 (distintas de la 2000 de la fase 1) y los hiperparámetros elegidos en el ajuste previo.
Los valores de k y de los orígenes se pasan por la línea de comandos para que el script siga la versión vigente del
preregistro (original: k = 20 y los 4 orígenes de validación; borrador de la enmienda 3: k = 10 y solo el último).
Todo se guarda fila a fila en resultados/e2/busqueda/{modelo}_fase2.csv y el script se reanuda donde quedó.

Uso: python src/fase2.py --k 10 --origenes ultimo GRU LSTM ...
     python src/fase2.py --k 20 --origenes todos MLP
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from origen_movil import leer_config, objetivos, pliegues, preparar_panel  # noqa: E402
from modelos_e2 import Datos, entrenar  # noqa: E402
from busqueda_arq import CARPETA, REF, anotar, registrar_hardware, ya_hechos  # noqa: E402

SEMILLAS = [2001, 2002, 2003, 2004, 2005]
EPOCAS, PACIENCIA = 50, 5
ID_REF = "w64-64-64_d010"


def finalistas(nombre, k):
    """Las k mejores de la fase 1 dentro de la muestra declarada, más la referencia (sin repetir)."""
    todas = ya_hechos(f"{CARPETA}/{nombre}_fase1.csv", "id")
    ruta_m = f"{CARPETA}/{nombre}_muestra_fase1.json"
    # MLP y CNN se evaluaron antes de que existiera el archivo de muestra: su fase 1 es completa (las 1 000)
    muestra = set(json.load(open(ruta_m))) if os.path.exists(ruta_m) else set(todas)
    filas = [r for r in todas.values() if r["id"] in muestra and r["estado"] == "ok"]
    faltan = len(muestra) - len(filas)
    if faltan:
        print(f"AVISO {nombre}: {faltan} arquitecturas de la muestra sin resultado válido en la fase 1", flush=True)
    filas.sort(key=lambda r: float(r["rmse_val"]))
    ids = [r["id"] for r in filas[:k]]
    if ID_REF not in ids:
        ids.append(ID_REF)
    return ids


def arquitectura(id_):
    w, b = id_[1:].split("_d")
    return dict(w=tuple(int(x) for x in w.split("-")), b=tuple(int(x) for x in b))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--origenes", choices=["ultimo", "todos"], required=True)
    ap.add_argument("--hasta", default=None, help="HH:MM: no empieza corridas nuevas después de esa hora (PC de noche)")
    ap.add_argument("modelos", nargs="+")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg = leer_config()
    p = preparar_panel(cfg)
    todos = pliegues(p, cfg, "validacion")
    elegidos = todos[-1:] if a.origenes == "ultimo" else todos
    for f in elegidos:
        datos = Datos(p, f)
        if "STGCN" in a.modelos:
            datos.activar_grafo(p, cfg["panel"])
        y_ev = objetivos(p, f["evaluacion"])[:, 0]
        for nombre in a.modelos:
            registrar_hardware(nombre, "fase 2")
            hp = json.load(open(f"{CARPETA}/{nombre}_hp_elegido.json"))
            ruta = f"{CARPETA}/{nombre}_fase2.csv"
            hechos = ya_hechos(ruta, "clave")
            for id_ in finalistas(nombre, a.k):
                for s in SEMILLAS:
                    clave = f"{id_}|{f['origen']}|{s}"
                    if clave in hechos:
                        continue
                    if a.hasta and a.hasta <= time.strftime("%H:%M") < "18:00":   # de noche no corta antes de medianoche
                        print(f"[{time.strftime('%H:%M')}] Hora de parada ({a.hasta}): se retoma en la próxima corrida")
                        sys.exit(0)
                    t0 = time.time()
                    try:
                        r = entrenar(nombre, datos, f, s, hp=arquitectura(id_), lr=hp["lr"], wd=hp["wd"],
                                     bs=hp["bs"], epocas=EPOCAS, paciencia=PACIENCIA)
                        err = r["pred"] - y_ev
                        rmse = float(np.sqrt(np.mean(err ** 2)))
                        ok = np.isfinite(rmse)
                        res = dict(rmse_val=rmse if ok else "", mae_val=float(np.mean(np.abs(err))) if ok else "",
                                   rmse_parada=r["rmse_parada"], rmse_train=r["historia_train"][-1],
                                   epocas=r["epocas"], segundos=round(r["segundos"], 1), parametros=r["parametros"],
                                   estado="ok" if ok else "excluida: NaN")
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        res = dict(rmse_val="", mae_val="", rmse_parada="", rmse_train="", epocas=0,
                                   segundos=round(time.time() - t0, 1), parametros="", estado="excluida: memoria GPU")
                    fila = dict(clave=clave, modelo=nombre, id=id_, origen=f["origen"], semilla=s,
                                referencia=int(id_ == ID_REF), k=a.k, **res)
                    anotar(ruta, fila)
                    print(f"[{time.strftime('%H:%M')}] {nombre} fase 2 {id_} {f['origen']} s{s} → "
                          f"{fila['rmse_val']} ({fila['segundos']} s)", flush=True)
