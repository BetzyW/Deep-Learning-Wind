"""Ajuste previo de hiperparámetros del modelo original (PADRE-TCN, diseño congelado V6), con el MISMO protocolo que
recibieron los benchmarks en el E2 (busqueda_arq.py, sección 14.2):
  * T = 20 configuraciones aleatorias de (lr, wd, lote), generadas con la semilla 7: es exactamente la misma lista
    de 20 configuraciones que probaron los demás modelos;
  * último origen de validación (2024-10), 10 % de las ventanas de ajuste, semilla 2000, 20 épocas, paciencia 3;
  * arquitectura fija: la del diseño congelado (TCN del E2 w = (64, 64, 16), b = (1, 1, 0); K = 6 anclas, pesos
    aprendidos, diferencia antisimétrica, λ = 0). No hay búsqueda de arquitectura: igual que los modelos publicados.
Se elige la configuración con menor RMSE de validación y se fija en resultados/e3/PADRE_configuracion.json, que
leen las corridas confirmatorias y las de los conjuntos adicionales. La prueba (2025) no se toca.
Salida: resultados/e3/PADRE_hp.csv (una fila por configuración; se reanuda).
Uso: python src/ajuste_padre.py [--hasta 06:00]      (dl_env, GPU)
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
RUTA = "resultados/e3/PADRE_hp.csv"
ARQ = dict(w=[64, 64, 16], b=[1, 1, 0], K=6, lam=0.0, antisimetrica=True, pesos="aprendidos")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hasta", default=None)
    a = ap.parse_args()
    import torch
    from origen_movil import leer_config, objetivos, pliegues, preparar_panel
    from modelos_e2 import Datos
    from busqueda_arq import anotar, correr, registrar_hardware, ya_hechos
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs("resultados/e3", exist_ok=True)
    cfg = leer_config()
    p = preparar_panel(cfg)
    f = pliegues(p, cfg, "validacion")[-1]
    datos = Datos(p, f)
    y_ev = objetivos(p, f["evaluacion"])[:, 0]
    rng = np.random.default_rng(7)          # idéntico a busqueda_arq.py
    espacio_hp = [dict(id=i, lr=float(10 ** rng.uniform(-4, np.log10(3e-3))), wd=float(rng.choice([0, 1e-5, 1e-4])),
                       bs=int(rng.choice([512, 1024, 2048]))) for i in range(20)]
    registrar_hardware("PADRE", "E3 ajuste previo")
    hechos = ya_hechos(RUTA, "id")
    arq = {**ARQ, "w": tuple(ARQ["w"]), "b": tuple(ARQ["b"])}
    for hp in espacio_hp:
        if str(hp["id"]) in hechos:
            continue
        if a.hasta and a.hasta <= time.strftime("%H:%M") < "18:00":
            print(f"[{time.strftime('%H:%M')}] Hora de parada ({a.hasta})", flush=True)
            sys.exit(0)
        fila = dict(modelo="PADRE", **hp, **correr("PADRE", datos, f, y_ev, 2000, arq, hp, 20),
                    fecha=time.strftime("%Y-%m-%d %H:%M"))
        anotar(RUTA, fila)
        print(f"[{time.strftime('%H:%M')}] PADRE hp {hp['id']:2d} lr={hp['lr']:.1e} wd={hp['wd']:.0e} "
              f"bs={hp['bs']} → {fila['rmse_val']} ({fila['segundos']} s)", flush=True)
    filas = [r for r in ya_hechos(RUTA, "id").values() if r["estado"] == "ok"]
    mejor = min(filas, key=lambda r: float(r["rmse_val"]))
    conf = dict(hp=dict(lr=float(mejor["lr"]), wd=float(mejor["wd"]), bs=int(mejor["bs"])), arquitectura=ARQ,
                rmse_val_ajuste=float(mejor["rmse_val"]), id_ajuste=int(mejor["id"]), diseno="V6 (design_log.md)",
                fijada=time.strftime("%Y-%m-%d %H:%M:%S"))
    json.dump(conf, open("resultados/e3/PADRE_configuracion.json", "w"), indent=1, ensure_ascii=False)
    print(f"Configuración fijada: {conf['hp']} (RMSE val {conf['rmse_val_ajuste']:.4f})", flush=True)
