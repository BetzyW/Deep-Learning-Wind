"""Validación con origen móvil de la selección de arquitectura (complemento de la enmienda 3; enmienda 8).

La enmienda 3 redujo la fase 2 a un origen de validación (2024-10-01). Este script comprueba si la elección habría
cambiado con los otros tres orígenes de 2024 (1-ene, 1-abr, 1-jul). Para cada modelo con búsqueda evalúa tres
finalistas de la fase 2, elegidos de antemano por regla y sin mirar la prueba:
  * la ELEGIDA (regla del preregistro: la de menos parámetros entre las empatadas a < δ/2 de la mejor),
  * la MEJOR por RMSE medio de la fase 2 (si es distinta de la elegida),
  * la PEOR de los finalistas por RMSE medio de la fase 2.
Si la elegida, la mejor y la peor quedan a menos de δ entre sí también en los otros orígenes, la elección con un solo
origen no cambia las conclusiones. Mismo protocolo que la fase 2: presupuesto completo (50 épocas, paciencia 5),
hiperparámetros del ajuste previo, semillas 2001 y 2002.
Salida: resultados/e2/busqueda/{modelo}_origenes.csv (fila a fila; se reanuda donde quedó).
Uso: python src/validacion_origenes.py MLP KAN CNN GRU LSTM TCN Transformer [--hasta 06:00]
     python src/validacion_origenes.py --resumen MLP ...      (tabla por origen; no usa GPU)
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
CARPETA = "resultados/e2/busqueda"

SEMILLAS = [2001, 2002]
EPOCAS, PACIENCIA = 50, 5
DELTA = 0.02


def tres_finalistas(nombre):
    import pandas as pd
    d = pd.read_csv(f"{CARPETA}/{nombre}_fase2.csv")
    d = d[d.estado == "ok"]
    r = d.groupby("id").agg(rmse=("rmse_val", "mean"), n=("rmse_val", "size"), parametros=("parametros", "first"))
    r = r[r.n >= 5].sort_values("rmse")
    empatadas = r[r.rmse < r.rmse.iloc[0] + DELTA / 2]
    elegida = empatadas.sort_values("parametros").index[0]
    papel = {elegida: "elegida"}
    papel.setdefault(r.index[0], "mejor")
    papel.setdefault(r.index[-1], "peor")
    return papel, r


def resumen(modelos):
    import pandas as pd
    filas = []
    for m in modelos:
        ruta = f"{CARPETA}/{m}_origenes.csv"
        if not os.path.exists(ruta):
            continue
        papel, r2 = tres_finalistas(m)
        d = pd.read_csv(ruta)
        d = d[d.estado == "ok"]
        f2 = pd.read_csv(f"{CARPETA}/{m}_fase2.csv")
        f2 = f2[(f2.estado == "ok") & f2.id.isin(papel) & f2.semilla.isin(SEMILLAS)]
        d = pd.concat([d, f2.assign(papel=f2.id.map(papel))[d.columns.intersection(f2.columns.union(["papel"]))]])
        t = d.groupby(["papel", "origen"]).rmse_val.mean().unstack()
        t["media_4_origenes"] = t.mean(1)
        for p_ in t.index:
            filas.append(dict(modelo=m, papel=p_, **t.loc[p_].round(4).to_dict()))
    out = pd.DataFrame(filas)
    print(out.to_string(index=False))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("modelos", nargs="+")
    ap.add_argument("--hasta", default=None, help="HH:MM: no empieza corridas nuevas después de esa hora (PC de noche)")
    ap.add_argument("--resumen", action="store_true")
    a = ap.parse_args()
    if a.resumen:
        resumen(a.modelos)
        sys.exit(0)
    import torch
    from origen_movil import leer_config, objetivos, pliegues, preparar_panel  # noqa: E402
    from modelos_e2 import Datos, entrenar  # noqa: E402
    from busqueda_arq import anotar, registrar_hardware, ya_hechos  # noqa: E402
    from fase2 import arquitectura  # noqa: E402
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg = leer_config()
    p = preparar_panel(cfg)
    origenes = pliegues(p, cfg, "validacion")[:-1]          # 1-ene, 1-abr y 1-jul de 2024 (el 1-oct ya está en la fase 2)
    for nombre in a.modelos:
        registrar_hardware(nombre, "validación con origen móvil")
        hp = json.load(open(f"{CARPETA}/{nombre}_hp_elegido.json"))
        papel, _ = tres_finalistas(nombre)
        print(f"{nombre}: {papel}", flush=True)
        ruta = f"{CARPETA}/{nombre}_origenes.csv"
        for f in origenes:
            hechos = ya_hechos(ruta, "clave")
            pend = [(i, s) for i in papel for s in SEMILLAS if f"{i}|{f['origen']}|{s}" not in hechos]
            if not pend:
                continue
            datos = Datos(p, f)
            y_ev = objetivos(p, f["evaluacion"])[:, 0]
            for id_, s in pend:
                if a.hasta and a.hasta <= time.strftime("%H:%M") < "18:00":
                    print(f"[{time.strftime('%H:%M')}] Hora de parada ({a.hasta}): se retoma en la próxima corrida")
                    sys.exit(0)
                t0 = time.time()
                try:
                    r = entrenar(nombre, datos, f, s, hp=arquitectura(id_), lr=hp["lr"], wd=hp["wd"], bs=hp["bs"],
                                 epocas=EPOCAS, paciencia=PACIENCIA)
                    err = r["pred"] - y_ev
                    rmse = float(np.sqrt(np.mean(err ** 2)))
                    ok = np.isfinite(rmse)
                    res = dict(rmse_val=rmse if ok else "", mae_val=float(np.mean(np.abs(err))) if ok else "",
                               rmse_parada=r["rmse_parada"], epocas=r["epocas"], segundos=round(r["segundos"], 1),
                               parametros=r["parametros"], estado="ok" if ok else "excluida: NaN")
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    res = dict(rmse_val="", mae_val="", rmse_parada="", epocas=0, segundos=round(time.time() - t0, 1),
                               parametros="", estado="excluida: memoria GPU")
                anotar(ruta, dict(clave=f"{id_}|{f['origen']}|{s}", modelo=nombre, id=id_, papel=papel[id_],
                                  origen=f["origen"], semilla=s, **res))
                print(f"[{time.strftime('%H:%M')}] {nombre} {papel[id_]} {id_} {f['origen']} s{s} → "
                      f"{res['rmse_val']} ({res['segundos']} s)", flush=True)
    print("Validación con origen móvil completa", flush=True)
