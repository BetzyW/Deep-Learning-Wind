"""Reproducción independiente de la tabla principal del E2 (Apéndice B del enunciado).

Dos niveles:
  1. `--tabla` (ml_env, CPU): recalcula el RMSE y el MAE de cada corrida final directamente desde las predicciones
     guardadas (.npz) y los datos del panel, por un camino de código distinto al de `finales.py` (que los calculó al
     entrenar), y reporta la discrepancia máxima frente a los CSV de las corridas.
  2. `--reentrenar` (dl_env, GPU): vuelve a entrenar desde cero corridas ya hechas en OTRO equipo (A100 de Colab) en
     el PC del proyecto (RTX 4060 Ti), con el mismo código, configuración y semilla, y compara el RMSE de prueba.
     Corridas elegidas antes de reentrenar: los 3 mejores modelos y el paper guía, semilla 1000, origen 2025-01-01
     (el más corto): Transformer, KAN, DLinear y MLP. TCN se corrió en el PC: se reentrena el mismo caso para medir
     la reproducibilidad en un mismo equipo.
Salida: resultados/e2/reproduccion/
Uso: python src/reproducir.py --tabla        |   python src/reproducir.py --reentrenar [--hasta HH:MM]
"""
import argparse, glob, json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
SALIDA = "resultados/e2/reproduccion"
CASOS = [("Transformer", "2025-01-01", 1000), ("KAN", "2025-01-01", 1000), ("DLinear", "2025-01-01", 1000),
         ("MLP", "2025-01-01", 1000), ("TCN", "2025-01-01", 1000)]


def tabla():
    from origen_movil import leer_config, preparar_panel
    p = preparar_panel(leer_config())
    y_todo = p.X[:, 0]
    filas = []
    for ruta in sorted(glob.glob("resultados/e2/finales/*.csv")):
        m = os.path.basename(ruta)[:-4]
        reg = pd.read_csv(ruta)
        for r in reg.itertuples():
            z = np.load(f"resultados/e2/finales/{m}/{r.origen}_{r.semilla}.npz")
            y = y_todo[z["tau"] + p.horizontes[0]]
            e = z["pred"].astype(np.float64) - y
            filas.append(dict(modelo=m, origen=r.origen, semilla=r.semilla, rmse_registrado=r.rmse,
                              rmse_recalculado=float(np.sqrt(np.nanmean(e ** 2))), mae_registrado=r.mae,
                              mae_recalculado=float(np.nanmean(np.abs(e)))))
    t = pd.DataFrame(filas)
    t["dif_rmse"] = (t.rmse_recalculado - t.rmse_registrado).abs()
    t["dif_mae"] = (t.mae_recalculado - t.mae_registrado).abs()
    t.to_csv(f"{SALIDA}/tabla_recalculada.csv", index=False)
    res = dict(corridas=int(len(t)), discrepancia_max_rmse=float(t.dif_rmse.max()),
               discrepancia_max_mae=float(t.dif_mae.max()))
    json.dump(res, open(f"{SALIDA}/tabla.json", "w"), indent=1)
    print(res)


def reentrenar(hasta):
    import torch
    from origen_movil import leer_config, objetivos, pliegues, preparar_panel
    from modelos_e2 import Datos, entrenar
    from fase2 import arquitectura
    from busqueda_arq import registrar_hardware
    cfg = leer_config()
    p = preparar_panel(cfg)
    ruta = f"{SALIDA}/reentrenadas.csv"
    hechas = set(pd.read_csv(ruta).clave) if os.path.exists(ruta) else set()
    for f in pliegues(p, cfg, "prueba"):
        casos = [c for c in CASOS if c[1] == f["origen"] and f"{c[0]}|{c[1]}|{c[2]}" not in hechas]
        if not casos:
            continue
        datos = Datos(p, f)
        y_ev = objetivos(p, f["evaluacion"])[:, 0]
        for m, o, s in casos:
            if hasta and hasta <= time.strftime("%H:%M") < "18:00":
                print(f"[{time.strftime('%H:%M')}] Hora de parada ({hasta})"); sys.exit(0)
            registrar_hardware(m, "reproducción")
            c = json.load(open(f"resultados/e2/finales/{m}_configuracion.json"))
            hp_arq = arquitectura(c["arquitectura"]["id"]) if c["arquitectura"] else None
            r = entrenar(m, datos, f, s, hp=hp_arq, lr=c["hp"]["lr"], wd=c["hp"]["wd"], bs=c["hp"]["bs"],
                         epocas=50, paciencia=5)
            orig = pd.read_csv(f"resultados/e2/finales/{m}.csv").set_index("clave").loc[f"{o}|{s}"]
            z = np.load(f"resultados/e2/finales/{m}/{o}_{s}.npz")
            fila = dict(clave=f"{m}|{o}|{s}", modelo=m, origen=o, semilla=s, gpu=torch.cuda.get_device_name(0),
                        rmse_original=float(orig.rmse), rmse_reentrenado=float(np.sqrt(np.nanmean((r["pred"] - y_ev) ** 2))),
                        epocas_original=int(orig.epocas), epocas_reentrenado=r["epocas"],
                        max_dif_prediccion=float(np.nanmax(np.abs(r["pred"] - z["pred"]))))
            fila["dif_rmse"] = fila["rmse_reentrenado"] - fila["rmse_original"]
            pd.DataFrame([fila]).to_csv(ruta, mode="a", header=not os.path.exists(ruta), index=False)
            print(f"[{time.strftime('%H:%M')}] {m} {o} s{s}: original {fila['rmse_original']:.4f}, "
                  f"reentrenado {fila['rmse_reentrenado']:.4f}", flush=True)
        del datos
        torch.cuda.empty_cache()
    print("Reproducción completa", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tabla", action="store_true")
    ap.add_argument("--reentrenar", action="store_true")
    ap.add_argument("--hasta", default=None)
    a = ap.parse_args()
    os.makedirs(SALIDA, exist_ok=True)
    if a.tabla:
        tabla()
    if a.reentrenar:
        reentrenar(a.hasta)
