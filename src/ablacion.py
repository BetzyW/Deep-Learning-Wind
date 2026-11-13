"""Estudio de ablación del E2 (sección 4.6 del enunciado; enmienda 7 del preregistro) sobre el mejor modelo, TCN
(primer lugar en el rango medio de Friedman entre estaciones), con su configuración final fija (w = (64, 64, 16),
b = (1, 1, 0), mismos hiperparámetros).

Componentes aislados, uno a la vez (el resto del modelo queda igual):
  * sin_dilatacion: convoluciones con dilatación 1 en los tres bloques (campo receptivo de 253 h a 37 h). Mide el
    aporte de la memoria larga, el elemento central de la TCN (Bai et al., 2018).
  * sin_residual: bloques sin conexión residual. Mide el aporte del camino residual.
  * sin_covariables: dirección (seno y coseno) y temperatura fijadas en su media de entrenamiento (0 tras el
    escalado); el modelo solo ve la velocidad. Mide el aporte de las variables exógenas.
Mismo protocolo que las corridas finales: los 4 orígenes de prueba de 2025, reentrenando en cada uno, semillas
declaradas 1000–1005 (R = 6). El modelo completo son las corridas finales de TCN, que ya existen.
Inferencia (con `--analizar`, en ml_env): por componente, Wilcoxon de rangos con signo sobre la diferencia de RMSE por
estación (media de semillas), Diebold–Mariano con HLN sobre la pérdida horaria y Holm sobre los 3 contrastes; además
media y desviación entre semillas por origen.
Uso: python src/ablacion.py [--hasta HH:MM]      (entrena, dl_env)
     python src/ablacion.py --analizar          (estadística, ml_env)
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

MODELO = "TCN"
ABLACIONES = {"sin_dilatacion": dict(dilatar=False), "sin_residual": dict(residual=False),
              "sin_covariables": dict(solo_vel=True)}
SEMILLAS = range(1000, 1006)
SALIDA = "resultados/e2/ablacion"


def entrenar_todo(hasta):
    import torch
    from origen_movil import leer_config, objetivos, pliegues, preparar_panel
    from modelos_e2 import Datos, entrenar
    from busqueda_arq import anotar, registrar_hardware, ya_hechos
    from fase2 import arquitectura
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    c = json.load(open(f"resultados/e2/finales/{MODELO}_configuracion.json"))
    hp_arq = arquitectura(c["arquitectura"]["id"])
    cfg = leer_config()
    p = preparar_panel(cfg)
    ruta = f"{SALIDA}/{MODELO}_ablacion.csv"
    os.makedirs(SALIDA, exist_ok=True)
    hechos = ya_hechos(ruta, "clave")
    registrar_hardware(MODELO, "ablación")
    for f in pliegues(p, cfg, "prueba"):
        datos = Datos(p, f)
        y_ev = objetivos(p, f["evaluacion"])[:, 0]
        for nombre, flags in ABLACIONES.items():
            os.makedirs(f"{SALIDA}/{nombre}", exist_ok=True)
            for s in SEMILLAS:
                clave = f"{nombre}|{f['origen']}|{s}"
                if clave in hechos:
                    continue
                if hasta and hasta <= time.strftime("%H:%M") < "18:00":
                    print(f"[{time.strftime('%H:%M')}] Hora de parada ({hasta}): se retoma en la próxima corrida")
                    sys.exit(0)
                r = entrenar(MODELO, datos, f, s, hp={**hp_arq, **flags}, lr=c["hp"]["lr"], wd=c["hp"]["wd"],
                             bs=c["hp"]["bs"], epocas=50, paciencia=5)
                np.savez_compressed(f"{SALIDA}/{nombre}/{f['origen']}_{s}.npz", tau=f["evaluacion"].astype(np.int32),
                                    pred=r["pred"].astype(np.float32))
                err = r["pred"] - y_ev
                fila = dict(clave=clave, ablacion=nombre, origen=f["origen"], semilla=s,
                            rmse=float(np.sqrt(np.nanmean(err ** 2))), mae=float(np.nanmean(np.abs(err))),
                            rmse_parada=r["rmse_parada"], epocas=r["epocas"], segundos=round(r["segundos"], 1),
                            parametros=r["parametros"])
                anotar(ruta, fila)
                print(f"[{time.strftime('%H:%M')}] {nombre} {f['origen']} s{s} → RMSE {fila['rmse']:.4f} "
                      f"({fila['segundos']} s)", flush=True)
        del datos
        torch.cuda.empty_cache()
    print("Ablación completa", flush=True)


def analizar():
    import pandas as pd
    from origen_movil import leer_config, preparar_panel
    from estadistica_e2 import cargar, diebold_mariano_hln, holm, wilcoxon_estaciones
    p = preparar_panel(leer_config())
    completo = cargar(p, MODELO).dropna(subset=["y"]).set_index("tau")
    filas, por_origen = [], []
    for nombre in ABLACIONES:
        a = cargar(p, nombre, carpeta=SALIDA).dropna(subset=["y"]).set_index("tau")
        comun = completo.index.intersection(a.index)
        c, b = completo.loc[comun], a.loc[comun]
        rmse_est = lambda d: d.assign(e2=(d.pred - d.y) ** 2).groupby("estacion").e2.mean().pipe(np.sqrt)
        w = wilcoxon_estaciones(rmse_est(b), rmse_est(c))
        err = lambda d: d.assign(e=d.pred - d.y).groupby("tiempo").e.mean()
        ec, eb = err(c), err(b)
        idx = ec.index.intersection(eb.index)
        dm = diebold_mariano_hln(eb.loc[idx].to_numpy(), ec.loc[idx].to_numpy())
        filas.append(dict(ablacion=nombre, n_ventanas=len(comun),
                          rmse_completo=float(np.sqrt(((c.pred - c.y) ** 2).mean())),
                          rmse_ablacion=float(np.sqrt(((b.pred - b.y) ** 2).mean())),
                          **{f"wilcoxon_{k}": v for k, v in w.items()}, dm_estadistico=dm["estadistico"], dm_p=dm["p"]))
    t = pd.DataFrame(filas)
    t["aumento_rmse"] = t.rmse_ablacion - t.rmse_completo
    t["wilcoxon_p_holm"] = holm(t.wilcoxon_p)
    t["dm_p_holm"] = holm(t.dm_p)
    t.to_csv(f"{SALIDA}/ablacion_inferencia.csv", index=False)
    s = pd.read_csv(f"{SALIDA}/{MODELO}_ablacion.csv")
    s = s[s.semilla.isin(SEMILLAS)]
    f = pd.read_csv(f"resultados/e2/finales/{MODELO}.csv")
    f = f[f.semilla.isin(SEMILLAS)].assign(ablacion="completo")
    resumen = (pd.concat([f, s]).groupby(["ablacion", "origen"]).rmse.agg(["mean", "std", "size"]).reset_index())
    resumen.to_csv(f"{SALIDA}/ablacion_por_origen.csv", index=False)
    print(t.round(4).to_string(index=False))
    print(resumen.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hasta", default=None)
    ap.add_argument("--analizar", action="store_true")
    a = ap.parse_args()
    analizar() if a.analizar else entrenar_todo(a.hasta)
