"""Ruta F2a del E2, parte con GPU: lo que necesita modelos entrenados (dl_env).

Modelos: los 3 mejores del benchmark según el rango medio de Friedman entre estaciones (la estación es la unidad de
independencia del preregistro): TCN, Transformer y KAN. Por RMSE agregado de todas las ventanas el tercero sería
Informer; se reporta en el capítulo, pero no se analiza aquí.

1. Residuos de validación para los intervalos conformes: cada modelo se reentrena con su configuración final en el
   último origen de VALIDACIÓN (2024-10-01) con las semillas declaradas 1000–1005 y se guardan sus predicciones del
   cuarto trimestre de 2024. La prueba no interviene en la calibración de los intervalos.
2. Modelos para estabilidad entre semillas: además del .pt guardado por `finales.py` (semilla 1000, origen 2025-10-01),
   se reentrenan las semillas 1001 y 1002 en ese mismo origen, con el mismo procedimiento, y se guardan.
3. Con los 3 modelos de cada arquitectura (semillas 1000–1002, origen 2025-10-01):
   * Razón de error fuera/dentro de muestra: RMSE de prueba del trimestre / RMSE en las ventanas de ajuste (modo
     evaluación, sin dropout).
   * Robustez: RMSE de prueba con la ENTRADA perturbada (ruido, horas faltantes, quiebre de nivel; 3 severidades).
   * Interpretabilidad: gradientes integrados en 2 000 ventanas de prueba fijas (semilla 0): importancia por hora de
     la ventana y por variable, prueba de fidelidad (borrar el 10 % de horas más importantes frente al 10 % al azar) y
     estabilidad entre semillas (Spearman entre perfiles de importancia).
Salida: resultados/e2/f2a/. Se reanuda: lo ya guardado no se repite.
Uso: python src/f2a_gpu.py [--hasta HH:MM]
"""
import argparse, itertools, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from origen_movil import leer_config, pliegues, preparar_panel  # noqa: E402
from modelos_e2 import MODELOS, Datos, entrenar, predecir  # noqa: E402
from fase2 import arquitectura  # noqa: E402
from ruta_f2a import SEVERIDADES, fidelidad_borrado, gradientes_integrados, perturbar  # noqa: E402

TOP3 = ["TCN", "Transformer", "KAN"]
SALIDA = "resultados/e2/f2a"
FINALES = "resultados/e2/finales"
SEM_VAL = range(1000, 1006)
SEM_PT = [1000, 1001, 1002]


def config(nombre):
    c = json.load(open(f"{FINALES}/{nombre}_configuracion.json"))
    return c, (arquitectura(c["arquitectura"]["id"]) if c["arquitectura"] else None)


def correr(nombre, datos, f, s, c, hp_arq):
    return entrenar(nombre, datos, f, s, hp=hp_arq, lr=c["hp"]["lr"], wd=c["hp"]["wd"], bs=c["hp"]["bs"],
                    epocas=50, paciencia=5)


def parar(hasta):
    if hasta and hasta <= time.strftime("%H:%M") < "18:00":
        print(f"[{time.strftime('%H:%M')}] Hora de parada ({hasta}): se retoma en la próxima corrida", flush=True)
        sys.exit(0)


def cargar_modelo(nombre, datos, ruta):
    c, hp_arq = config(nombre)
    m = MODELOS[nombre](datos.L, datos.d, **(hp_arq or {})).to(datos.dev)
    m.load_state_dict(torch.load(ruta, map_location=datos.dev)["estado"])
    return m.eval()


@torch.no_grad()
def rmse_perturbado(m, datos, idx, y, tipo, nivel, semilla=0, bs=4096):
    g = torch.Generator(device=datos.dev).manual_seed(semilla)
    pred = []
    for b in np.array_split(idx, max(1, len(idx) // bs)):
        x, _ = datos(b)
        pred.append(m(perturbar(x, tipo, nivel, g) if tipo else x).float().cpu().numpy())
    return float(np.sqrt(np.mean((np.concatenate(pred) - y) ** 2)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hasta", default=None)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg = leer_config()
    p = preparar_panel(cfg)
    f_val = pliegues(p, cfg, "validacion")[-1]
    f_pru = pliegues(p, cfg, "prueba")[-1]
    assert f_val["origen"] == "2024-10-01" and f_pru["origen"] == "2025-10-01"

    # 1) residuos de validación
    datos = Datos(p, f_val)
    for nombre in TOP3:
        c, hp_arq = config(nombre)
        os.makedirs(f"{SALIDA}/{nombre}", exist_ok=True)
        for s in SEM_VAL:
            ruta = f"{SALIDA}/{nombre}/validacion_{f_val['origen']}_{s}.npz"
            if os.path.exists(ruta):
                continue
            parar(a.hasta)
            r = correr(nombre, datos, f_val, s, c, hp_arq)
            np.savez_compressed(ruta, tau=f_val["evaluacion"].astype(np.int32), pred=r["pred"].astype(np.float32))
            print(f"[{time.strftime('%H:%M')}] {nombre} validación {f_val['origen']} s{s} ({r['segundos']:.0f} s)",
                  flush=True)
    del datos
    torch.cuda.empty_cache()

    # 2) modelos de las semillas 1001 y 1002 en el último origen de prueba
    datos = Datos(p, f_pru)
    for nombre in TOP3:
        c, hp_arq = config(nombre)
        for s in SEM_PT[1:]:
            ruta = f"{SALIDA}/{nombre}/{f_pru['origen']}_{s}.pt"
            if os.path.exists(ruta):
                continue
            parar(a.hasta)
            r = correr(nombre, datos, f_pru, s, c, hp_arq)
            torch.save(dict(estado=r["estado"], hp=c, semilla=s, origen=f_pru["origen"]), ruta)
            print(f"[{time.strftime('%H:%M')}] {nombre} prueba {f_pru['origen']} s{s} guardado ({r['segundos']:.0f} s)",
                  flush=True)

    # 3) análisis con los 3 modelos de cada arquitectura
    ruta_res = f"{SALIDA}/gpu_resultados.json"
    if os.path.exists(ruta_res):
        print("Análisis ya hecho:", ruta_res)
        sys.exit(0)
    ev, aj = f_pru["evaluacion"], f_pru["ajuste"]
    y_ev = datos.y[torch.as_tensor(ev + datos.h, device=datos.dev)].cpu().numpy()
    y_aj = datos.y[torch.as_tensor(aj + datos.h, device=datos.dev)].cpu().numpy()
    rng = np.random.default_rng(0)
    muestra = np.sort(rng.choice(ev, 2000, replace=False))
    res = dict(modelos=TOP3, origen=f_pru["origen"], semillas=SEM_PT, variables=list(p.cols), L=p.L,
               fuera_dentro={}, robustez={}, interpretabilidad={})
    for nombre in TOP3:
        perfiles_h, perfiles_v, fid = [], [], []
        for s in SEM_PT:
            ruta = (f"{FINALES}/{nombre}/{f_pru['origen']}_{s}.pt" if s == SEM_PT[0]
                    else f"{SALIDA}/{nombre}/{f_pru['origen']}_{s}.pt")
            m = cargar_modelo(nombre, datos, ruta)
            fuera = float(np.sqrt(np.mean((predecir(m, datos, ev) - y_ev) ** 2)))
            dentro = float(np.sqrt(np.mean((predecir(m, datos, aj) - y_aj) ** 2)))
            res["fuera_dentro"].setdefault(nombre, []).append(dict(semilla=s, rmse_prueba=fuera, rmse_ajuste=dentro,
                                                                   razon=fuera / dentro))
            filas = [dict(semilla=s, tipo="ninguna", nivel=0.0, rmse=fuera)]
            for tipo, niveles in SEVERIDADES.items():
                for nivel in niveles:
                    filas.append(dict(semilla=s, tipo=tipo, nivel=nivel,
                                      rmse=rmse_perturbado(m, datos, ev, y_ev, tipo, nivel, semilla=s)))
            res["robustez"].setdefault(nombre, []).extend(filas)
            # gradientes integrados por lotes de 250 ventanas
            g = torch.Generator(device=datos.dev).manual_seed(0)
            atr_h, atr_v, f_top, f_azar = np.zeros(p.L), np.zeros(len(p.cols)), [], []
            for b in np.array_split(muestra, 8):
                x, _ = datos(b)
                at = gradientes_integrados(m, x)
                atr_h += at.abs().sum(-1).sum(0).cpu().numpy()
                atr_v += at.abs().sum(1).sum(0).cpu().numpy()
                fb = fidelidad_borrado(m, x, at, rng=g)
                f_top.append(fb["cambio_top"]); f_azar.append(fb["cambio_azar"])
            perfiles_h.append(atr_h / atr_h.sum()); perfiles_v.append(atr_v / atr_v.sum())
            fid.append(dict(semilla=s, cambio_top=float(np.mean(f_top)), cambio_azar=float(np.mean(f_azar))))
            print(f"[{time.strftime('%H:%M')}] {nombre} s{s}: fuera/dentro {fuera / dentro:.3f}, "
                  f"fidelidad {np.mean(f_top):.3f} vs {np.mean(f_azar):.3f}", flush=True)
            del m
            torch.cuda.empty_cache()
        from scipy.stats import spearmanr
        estab_h = [float(spearmanr(perfiles_h[i], perfiles_h[j])[0]) for i, j in itertools.combinations(range(3), 2)]
        estab_v = [float(spearmanr(perfiles_v[i], perfiles_v[j])[0]) for i, j in itertools.combinations(range(3), 2)]
        res["interpretabilidad"][nombre] = dict(
            importancia_hora=np.mean(perfiles_h, 0).tolist(), importancia_variable=np.mean(perfiles_v, 0).tolist(),
            importancia_hora_por_semilla=[x.tolist() for x in perfiles_h], fidelidad=fid,
            spearman_horas=estab_h, spearman_variables=estab_v)
    json.dump(res, open(ruta_res, "w"), indent=1, ensure_ascii=False)
    print("Listo:", ruta_res, flush=True)
