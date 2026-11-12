"""Corridas finales del E2 en los 4 orígenes de PRUEBA de 2025 (sección 6 del preregistro; enmienda 3, punto 5).

Para cada modelo:
  * Configuración: hiperparámetros del ajuste previo (`{modelo}_hp_elegido.json`) y, si tuvo búsqueda de arquitectura,
    la ganadora de la fase 2 con la regla declarada: menor RMSE de validación medio sobre las 5 semillas; si otra
    arquitectura queda a menos de δ/2 = 0.01, gana la de menos parámetros. Modelos con arquitectura publicada usan la
    suya. La elección se hace solo con validación y se guarda antes de tocar la prueba.
  * En cada origen de prueba se reentrena desde cero (entrenamiento + parada del origen), máximo 50 épocas, paciencia 5,
    y se pronostican las ventanas de ese trimestre.
  * Semillas: 1000–1005 son las declaradas para el E2 (R = 6); 1006–1009 se agregan para el E3 (≥ 10 semillas) y se
    reportan aparte.
Se guardan, por corrida: métricas (resultados/e2/finales/{modelo}.csv), predicciones por ventana (.npz, para DM, MCS,
Wilcoxon por estación, etc.) y un modelo entrenado por arquitectura (.pt: primera semilla, último origen), para
exportarlo a ONNX en el E3. Se reanuda donde quedó.
Uso: python src/finales.py Transformer KAN ...   [--semillas 1000-1009]
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from origen_movil import leer_config, objetivos, pliegues, preparar_panel  # noqa: E402
from modelos_e2 import Datos, entrenar  # noqa: E402
from busqueda_arq import CARPETA, PUBLICADOS, anotar, registrar_hardware, ya_hechos  # noqa: E402
from fase2 import arquitectura  # noqa: E402

SALIDA = "resultados/e2/finales"
DELTA = 0.02


def ganadora(nombre):
    """Arquitectura ganadora de la fase 2 (regla del preregistro) o None si el modelo tiene arquitectura publicada."""
    if nombre in PUBLICADOS:
        return None
    import pandas as pd
    d = pd.read_csv(f"{CARPETA}/{nombre}_fase2.csv")
    d = d[d.estado == "ok"]
    r = d.groupby("id").agg(rmse=("rmse_val", "mean"), n=("rmse_val", "size"), parametros=("parametros", "first"))
    r = r[r.n >= 5].sort_values("rmse")
    empatadas = r[r.rmse < r.rmse.iloc[0] + DELTA / 2]
    elegida = empatadas.sort_values("parametros").index[0]
    return dict(id=elegida, rmse_val_fase2=float(r.loc[elegida, "rmse"]), mejor_id=r.index[0],
                empatadas=list(empatadas.index))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("modelos", nargs="+")
    ap.add_argument("--semillas", default="1000-1009")
    ap.add_argument("--hasta", default=None, help="HH:MM: no empieza corridas nuevas después de esa hora (PC de noche)")
    a = ap.parse_args()
    s0, s1 = map(int, a.semillas.split("-"))
    semillas = list(range(s0, s1 + 1))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(SALIDA, exist_ok=True)
    # 1) configuración de cada modelo, fijada con validación ANTES de abrir la prueba
    conf = {}
    for nombre in a.modelos:
        ruta_c = f"{SALIDA}/{nombre}_configuracion.json"
        if os.path.exists(ruta_c):
            conf[nombre] = json.load(open(ruta_c))
            continue
        g = ganadora(nombre)
        conf[nombre] = dict(hp=json.load(open(f"{CARPETA}/{nombre}_hp_elegido.json")), arquitectura=g,
                            fijada=time.strftime("%Y-%m-%d %H:%M:%S"))
        json.dump(conf[nombre], open(ruta_c, "w"), indent=1)
        print(f"{nombre}: configuración fijada {json.dumps(conf[nombre], ensure_ascii=False)}", flush=True)
    # 2) corridas en los orígenes de prueba
    cfg = leer_config()
    p = preparar_panel(cfg)
    for f in pliegues(p, cfg, "prueba"):
        datos = Datos(p, f)
        if "STGCN" in a.modelos:
            datos.activar_grafo(p, cfg["panel"])
        y_ev = objetivos(p, f["evaluacion"])[:, 0]
        for nombre in a.modelos:
            registrar_hardware(nombre, "corridas finales")
            c = conf[nombre]
            hp_arq = arquitectura(c["arquitectura"]["id"]) if c["arquitectura"] else None
            ruta = f"{SALIDA}/{nombre}.csv"
            hechos = ya_hechos(ruta, "clave")
            os.makedirs(f"{SALIDA}/{nombre}", exist_ok=True)
            for s in semillas:
                clave = f"{f['origen']}|{s}"
                if clave in hechos:
                    continue
                if a.hasta and a.hasta <= time.strftime("%H:%M") < "18:00":   # de noche no corta antes de medianoche
                    print(f"[{time.strftime('%H:%M')}] Hora de parada ({a.hasta}): se retoma en la próxima corrida")
                    sys.exit(0)
                r = entrenar(nombre, datos, f, s, hp=hp_arq, lr=c["hp"]["lr"], wd=c["hp"]["wd"], bs=c["hp"]["bs"],
                             epocas=50, paciencia=5)
                err = r["pred"] - y_ev
                base = f"{SALIDA}/{nombre}/{f['origen']}_{s}"
                np.savez_compressed(base + ".npz", tau=f["evaluacion"].astype(np.int32), pred=r["pred"].astype(np.float32))
                if s == semillas[0] and f["origen"] == cfg["origenes_prueba"][-1]:
                    # un solo modelo guardado por arquitectura (primera semilla, último origen: el que más datos vio),
                    # que es el que se exporta en el E3; así el Drive no se llena con 40 copias por modelo
                    torch.save(dict(estado=r["estado"], hp=c, semilla=s, origen=f["origen"]), base + ".pt")
                fila = dict(clave=clave, modelo=nombre, origen=f["origen"], semilla=s, declarada_e2=int(s <= 1005),
                            rmse=float(np.sqrt(np.mean(err ** 2))), mae=float(np.mean(np.abs(err))),
                            rmse_parada=r["rmse_parada"], rmse_train=r["historia_train"][-1], epocas=r["epocas"],
                            segundos=round(r["segundos"], 1), parametros=r["parametros"])
                anotar(ruta, fila)
                print(f"[{time.strftime('%H:%M')}] {nombre} prueba {f['origen']} s{s} → RMSE {fila['rmse']:.4f} "
                      f"({fila['segundos']} s)", flush=True)
