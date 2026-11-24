"""Corridas confirmatorias del modelo original (PADRE-TCN) y de su ablación, según experiments/protocolo.md
(tag e3-protocolo). Mismo procedimiento que las corridas finales de los benchmarks (`finales.py`):
  * los 4 orígenes de prueba de 2025, reentrenando desde cero en cada uno (entrenamiento + parada del origen);
  * semillas 1000–1009 (pares estrictos con los benchmarks), máximo 50 épocas, paciencia 5;
  * hiperparámetros fijados en resultados/e3/PADRE_configuracion.json (ajuste previo con T = 20, origen 2024-10).
Variantes: V6 es el diseño congelado (protocolo, sección 2). La ablación (protocolo, sección 11) cambia un componente:
  V3 = K = 1 · V4 = pesos uniformes · V5 = diferencia sin antisimetría · V1 = pérdida por ancla λ = 0.5.
Por corrida se guarda una fila en resultados/e3/confirmatorias/{variante}.csv y, por ventana, el pronóstico y la
dispersión entre anclas en {variante}/{origen}_{semilla}.npz (para los intervalos conformales y las pruebas). El modelo
de la primera semilla en el último origen se guarda en .pt para exportarlo a ONNX. Se reanuda donde quedó.
Uso: python src/confirmatorias_padre.py [V6 V3 V4 V5 V1] [--semillas 1000-1009] [--hasta 06:00]   (dl_env, GPU)
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from origen_movil import leer_config, objetivos, pliegues, preparar_panel  # noqa: E402
from modelos_e2 import Datos, entrenar  # noqa: E402
from busqueda_arq import anotar, registrar_hardware, ya_hechos  # noqa: E402

SALIDA = "resultados/e3/confirmatorias"
VARIANTES = {          # cambios respecto del diseño congelado V6 (ver design_log.md)
    "V6": {},
    "V3": dict(K=1),
    "V4": dict(pesos="uniformes"),
    "V5": dict(antisimetrica=False),
    "V1": dict(lam=0.5),
}


@torch.no_grad()
def dispersion(r, datos, ev, arq):
    from original_model import PadreTCN
    m = PadreTCN(datos.L, datos.d, **arq, mu_y=datos.mu_y, sd_y=datos.sd_y).to(datos.dev).eval()
    m.load_state_dict(r["estado"])
    return np.concatenate([m.dispersion(datos(ev[i:i + 4096])[0]).float().cpu().numpy()
                           for i in range(0, len(ev), 4096)])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("variantes", nargs="*", default=["V6"])
    ap.add_argument("--semillas", default="1000-1009")
    ap.add_argument("--hasta", default=None, help="HH:MM: no empieza corridas nuevas después de esa hora (PC de noche)")
    a = ap.parse_args()
    s0, s1 = map(int, a.semillas.split("-"))
    semillas = list(range(s0, s1 + 1))
    conf = json.load(open("resultados/e3/PADRE_configuracion.json"))
    hp, base_arq = conf["hp"], conf["arquitectura"]
    base_arq = {**base_arq, "w": tuple(base_arq["w"]), "b": tuple(base_arq["b"])}
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(SALIDA, exist_ok=True)
    cfg = leer_config()
    p = preparar_panel(cfg)
    for f in pliegues(p, cfg, "prueba"):
        ev = np.asarray(f["evaluacion"])
        pend = [(v, s) for v in a.variantes for s in semillas
                if f"{f['origen']}|{s}" not in ya_hechos(f"{SALIDA}/{v}.csv", "clave")]
        if not pend:
            continue
        datos = Datos(p, f)
        y_ev = objetivos(p, ev)[:, 0]
        for v, s in pend:
            if a.hasta and a.hasta <= time.strftime("%H:%M") < "18:00":   # de noche no corta antes de medianoche
                print(f"[{time.strftime('%H:%M')}] Hora de parada ({a.hasta}): se retoma en la próxima corrida", flush=True)
                sys.exit(0)
            arq = {**base_arq, **VARIANTES[v]}
            registrar_hardware("PADRE", f"E3 confirmatorias {v}")
            r = entrenar("PADRE", datos, f, s, hp=arq, lr=hp["lr"], wd=hp["wd"], bs=hp["bs"], epocas=50, paciencia=5)
            disp = dispersion(r, datos, ev, arq)
            os.makedirs(f"{SALIDA}/{v}", exist_ok=True)
            ruta = f"{SALIDA}/{v}/{f['origen']}_{s}"
            np.savez_compressed(ruta + ".npz", tau=ev.astype(np.int32), pred=r["pred"].astype(np.float32),
                                disp=disp.astype(np.float32))
            if s == semillas[0] and f["origen"] == cfg["origenes_prueba"][-1]:
                torch.save(dict(estado=r["estado"], hp=hp, arquitectura=arq, semilla=s, origen=f["origen"]),
                           ruta + ".pt")
            err = r["pred"] - y_ev
            fila = dict(clave=f"{f['origen']}|{s}", variante=v, origen=f["origen"], semilla=s,
                        rmse=float(np.sqrt(np.mean(err ** 2))), mae=float(np.mean(np.abs(err))),
                        rmse_parada=r["rmse_parada"], rmse_train=r["historia_train"][-1], epocas=r["epocas"],
                        segundos=round(r["segundos"], 1), parametros=r["parametros"],
                        fecha=time.strftime("%Y-%m-%d %H:%M"))
            anotar(f"{SALIDA}/{v}.csv", fila)
            print(f"[{time.strftime('%H:%M')}] {v} prueba {f['origen']} s{s} → RMSE {fila['rmse']:.4f} "
                  f"({fila['segundos']} s)", flush=True)
        del datos
        torch.cuda.empty_cache()
    print("Confirmatorias: completo", flush=True)
