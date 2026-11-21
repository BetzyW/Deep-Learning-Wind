"""Corridas de DISEÑO del modelo original (sección 7.2 del enunciado del E3): partición de desarrollo.

Desarrollo = los tres primeros orígenes de validación (2024-01, 2024-04 y 2024-07). No se usa el origen 2024-10 (que
queda para el ajuste de hiperparámetros, igual que en los benchmarks) ni la prueba de 2025. Cada variante se anota en
experiments/design_log.md con su métrica de desarrollo; el número total de variantes es N_d.
Base fija durante el diseño: arquitectura de la TCN elegida en el E2 (w = (64, 64, 16), b = (1, 1, 0)) y sus
hiperparámetros (lr, wd, lote); semilla 2001; mismo presupuesto (50 épocas, paciencia 5).
Ronda 2 (--ronda 2): las mismas variantes más la TCN de referencia (V0), guardando por ventana el pronóstico y la
dispersión entre anclas en resultados/e3/diseno_r2/{variante}_{origen}.npz, para medir el sesgo con viento ≥ 6 m/s y
la relación entre dispersión y error (la hipótesis del mecanismo).
Salida: resultados/e3/diseno.csv o diseno_r2.csv (una fila por variante y origen; se reanuda)
Uso: python src/diseno_padre.py [--ronda 2] [--hasta 06:00]      (dl_env, GPU)
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
RUTA = "resultados/e3/diseno.csv"
SEMILLA = 2001
VARIANTES = {   # id: (descripción, hiperparámetros propios)
    "V1": ("completo: K = 6, λ = 0.5, antisimétrica, pesos aprendidos", dict(K=6, lam=0.5)),
    "V2": ("K = 3 anclas", dict(K=3, lam=0.5)),
    "V3": ("K = 1: solo el ancla de persistencia (corrección sobre la persistencia)", dict(K=1, lam=0.5)),
    "V4": ("pesos uniformes entre anclas", dict(K=6, lam=0.5, pesos="uniformes")),
    "V5": ("diferencia sin antisimetría", dict(K=6, lam=0.5, antisimetrica=False)),
    "V6": ("sin pérdida por ancla (λ = 0)", dict(K=6, lam=0.0)),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hasta", default=None)
    ap.add_argument("--ronda", type=int, default=1)
    a = ap.parse_args()
    if a.ronda == 2:
        RUTA = "resultados/e3/diseno_r2.csv"
        VARIANTES = {"V0": ("TCN del E2 (referencia, sin componente propio)", None), **VARIANTES}
        os.makedirs("resultados/e3/diseno_r2", exist_ok=True)
    import torch
    from origen_movil import leer_config, objetivos, pliegues, preparar_panel
    from modelos_e2 import Datos, entrenar
    from busqueda_arq import anotar, registrar_hardware, ya_hechos
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs("resultados/e3", exist_ok=True)
    hp = json.load(open("resultados/e2/busqueda/TCN_hp_elegido.json"))
    arq = dict(w=(64, 64, 16), b=(1, 1, 0))
    cfg = leer_config()
    p = preparar_panel(cfg)
    registrar_hardware("PADRE", "diseño (desarrollo)")
    for f in pliegues(p, cfg, "validacion")[:3]:
        hechos = ya_hechos(RUTA, "clave")
        pend = [v for v in VARIANTES if f"{v}|{f['origen']}" not in hechos]
        if not pend:
            continue
        datos = Datos(p, f)
        y_ev = objetivos(p, f["evaluacion"])[:, 0]
        for v in pend:
            if a.hasta and a.hasta <= time.strftime("%H:%M") < "18:00":
                print(f"[{time.strftime('%H:%M')}] Hora de parada ({a.hasta})", flush=True)
                sys.exit(0)
            desc, extra = VARIANTES[v]
            nombre, hp_m = ("TCN", arq) if extra is None else ("PADRE", {**arq, **extra})
            r = entrenar(nombre, datos, f, SEMILLA, hp=hp_m, lr=hp["lr"], wd=hp["wd"], bs=hp["bs"],
                         epocas=50, paciencia=5)
            err = r["pred"] - y_ev
            if a.ronda == 2:
                disp = np.full(len(y_ev), np.nan, dtype=np.float32)
                if nombre == "PADRE":
                    from original_model import PadreTCN
                    m = PadreTCN(datos.L, datos.d, **hp_m, mu_y=datos.mu_y, sd_y=datos.sd_y).to(datos.dev).eval()
                    m.load_state_dict(r["estado"])
                    ev = np.asarray(f["evaluacion"])
                    disp = np.concatenate([m.dispersion(datos(ev[i:i + 4096])[0]).cpu().numpy()
                                           for i in range(0, len(ev), 4096)])
                np.savez_compressed(f"resultados/e3/diseno_r2/{v}_{f['origen']}.npz", tau=np.asarray(f["evaluacion"]),
                                    pred=r["pred"].astype(np.float32), y=y_ev.astype(np.float32), disp=disp)
            fila = dict(clave=f"{v}|{f['origen']}", variante=v, descripcion=desc, origen=f["origen"],
                        semilla=SEMILLA, rmse_val=float(np.sqrt(np.mean(err ** 2))),
                        mae_val=float(np.mean(np.abs(err))), rmse_parada=r["rmse_parada"], epocas=r["epocas"],
                        segundos=round(r["segundos"], 1), parametros=r["parametros"],
                        fecha=time.strftime("%Y-%m-%d %H:%M"))
            anotar(RUTA, fila)
            print(f"[{time.strftime('%H:%M')}] {v} {f['origen']} → {fila['rmse_val']:.4f} ({fila['segundos']} s)",
                  flush=True)
    print("Diseño completo", flush=True)
