"""Complementos de timeseries-cv (sección 2.21 del curso). La reducción de los datos fue indicación del profesor Rubio
(10-nov): una sola estación, la de menos faltantes (Los Garzones), y las últimas ≈ 12 000 h, para que el cómputo
quepa. La búsqueda ya está en resultados/tsxv/busqueda.csv; este script agrega:

1. Top 3 por entrada (24, 48, 96 y 192 h): menor RMSE de validación entre las que pasan gap_ratio < 0.15. Con 192 h
   ninguna pasa, así que se toman las 3 de menor gap_ratio y se reporta así.
2. Cada una se reentrena con su misma semilla hasta su época de mejor validación (la trayectoria aleatoria es la misma,
   así que el modelo es el mismo de la búsqueda) y se guardan las predicciones de validación de cada pliegue, para
   MAE, RMSE y MAPE por pliegue y BDS de los residuos. Las ventanas de validación de pliegues consecutivos son bloques
   de 24 h contiguos, así que sus residuos forman una serie horaria continua (salvo huecos).
3. La ganadora de cada entrada se reevalúa con K-Fold (k = 5, barajado, semilla 0) y con Group K-Fold (k = 5, grupos =
   meses calendario contiguos), con la misma receta. K-Fold mezcla pasado y futuro y deja ventanas solapadas a ambos
   lados (se espera que sea optimista); Group K-Fold por meses reduce ese efecto.
Salida: resultados/tsxv/extras/ (I{entrada}_top{k}.npz y kfold.csv).
Uso: python src/tsxv_extras.py          (dl_env, GPU; minutos)
"""
import json, os, sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from timeseries_cv import Lotes, completas, entrenar_configuracion, leer_serie, problema_cv, rejilla, ventanas  # noqa

SAL = "resultados/tsxv/extras"


def top3(d, I):
    g = d[d.entrada == I]
    ok = g[g.gap_ratio < 0.15].sort_values("best_val_rmse")
    return (ok if len(ok) else g.sort_values("gap_ratio")).head(3)


def capas_mascara(arq):
    w, m = arq[1:].split("_d")
    return tuple(int(x) for x in w.split("-")), tuple(int(x) for x in m)


if __name__ == "__main__":
    cfg = json.load(open("configs/tsxv.json", encoding="utf-8"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(SAL, exist_ok=True)
    d = pd.read_csv(f"{cfg['salida']}/busqueda.csv")
    s = leer_serie(cfg)
    v = s.to_numpy(np.float64)
    O, J = cfg["horizonte"], cfg["salto"]
    n_hold = max(int(round(len(v) * cfg["holdout_fraccion"])), 2 * O)
    n_cv = len(v) - n_hold
    configs = list(rejilla(cfg))
    filas_k = []
    for I in cfg["entradas"]:
        P = problema_cv(v, n_cv, I, O, J, dev)
        completa = completas(v, I, O)
        for rango, r in enumerate(top3(d, I).itertuples(), 1):
            capas, mascara = capas_mascara(r.arquitectura)
            semilla = cfg["semilla_base"] + configs.index((capas, mascara))
            c = dict(cfg, epocas=int(r.epoca_best))
            _, _, cv_ms, pred = entrenar_configuracion(P, capas, mascara, c, semilla, dev, con_pred=True)
            np.savez_compressed(f"{SAL}/I{I}_top{rango}.npz", pred=pred[:, 0], y=P.Ycv_crudo[:, 0],
                                arquitectura=r.arquitectura, epoca=int(r.epoca_best), gap_ratio=float(r.gap_ratio),
                                rmse_busqueda=float(r.best_val_rmse_ms), rmse_repetido=float(cv_ms[-1].mean()))
            print(f"I={I} top{rango} {r.arquitectura}: RMSE val {cv_ms[-1].mean():.4f} m/s "
                  f"(en la búsqueda {r.best_val_rmse_ms:.4f}), gap_ratio {r.gap_ratio:.3f}", flush=True)
            if rango > 1:
                continue
            # K-Fold y Group K-Fold sobre las ventanas válidas del tramo de validación cruzada (paso J)
            todas = J * np.arange((n_cv - I - O) // J + 1)
            todas = todas[completa(todas)]
            X, Y = ventanas(v, todas, I, O)
            mes = s.index[todas + I].to_period("M").astype(str).to_numpy()
            kf = np.array_split(np.random.default_rng(0).permutation(len(todas)), 5)
            gk = [np.flatnonzero(np.isin(mes, g)) for g in np.array_split(np.unique(mes), 5)]
            for esquema, pl in [("K-Fold", kf), ("Group K-Fold (meses)", gk)]:
                rm = []
                for k, val in enumerate(pl):
                    trn = np.setdiff1d(np.arange(len(todas)), val)
                    L = Lotes(X[trn], Y[trn], np.array([len(trn)]), X[val][None], Y[val][None], dev)
                    _, cvk, cvk_ms = entrenar_configuracion(L, capas, mascara, cfg, semilla, dev)
                    e = int(np.argmin(cvk))
                    rm.append(float(cvk_ms[e, 0]))
                    filas_k.append(dict(entrada=I, arquitectura=r.arquitectura, esquema=esquema, pliegue=k + 1,
                                        n_entrenamiento=len(trn), n_validacion=len(val), rmse_ms=rm[-1],
                                        epoca_mejor=e + 1))
                print(f"  {esquema}: RMSE medio {np.mean(rm):.4f} m/s (DE {np.std(rm):.4f})", flush=True)
    pd.DataFrame(filas_k).to_csv(f"{SAL}/kfold.csv", index=False)
    print("Extras de timeseries-cv completos", flush=True)
