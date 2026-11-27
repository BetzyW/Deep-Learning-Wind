"""Interpretabilidad y robustez del modelo original (PADRE-TCN, V6), medidas exactamente igual que las de los mejores
benchmarks en la ruta F2a del E2 (`f2a_gpu.py`), para poder compararlas.

1. Modelos: el .pt de la semilla 1000 en el origen 2025-10-01 (confirmatorias) y las semillas 1001 y 1002 en ese mismo
   origen, reentrenadas con el mismo procedimiento. Se comprueba que sus predicciones coinciden con las guardadas en
   las corridas confirmatorias (reproducibilidad) y se guardan en resultados/e3/interpretabilidad/.
2. Gradientes integrados (Sundararajan et al., 2017; línea base 0, 32 pasos) en las MISMAS 2 000 ventanas de prueba que
   el E2 (semilla 0): importancia por hora de la ventana y por variable, fidelidad por borrado (borrar el 10 % de horas
   más importantes frente al 10 % al azar) y estabilidad entre semillas (Spearman entre perfiles).
3. Interpretabilidad propia del mecanismo, en todas las ventanas de prueba del origen: peso medio de cada ancla
   (día j = 1, …, 6), en total y con viento observado ≥ 6 m/s, y frecuencia con que cada ancla es la de mayor peso.
4. Robustez: RMSE con la entrada perturbada (ruido, horas faltantes y quiebre de nivel, 3 severidades cada una) y razón
   de error fuera/dentro de muestra, como en el E2.
Salida: resultados/e3/interpretabilidad/resultados.json
Uso: python src/interpretabilidad_padre.py      (dl_env, GPU; unos 10 min)
"""
import itertools, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from origen_movil import leer_config, pliegues, preparar_panel  # noqa: E402
from modelos_e2 import Datos, entrenar, predecir  # noqa: E402
from original_model import PadreTCN  # noqa: E402
from ruta_f2a import SEVERIDADES, fidelidad_borrado, gradientes_integrados  # noqa: E402
from f2a_gpu import rmse_perturbado  # noqa: E402

SALIDA = "resultados/e3/interpretabilidad"
CONF = "resultados/e3/confirmatorias/V6"
SEMILLAS = [1000, 1001, 1002]


def construir(datos, arq, estado=None):
    m = PadreTCN(datos.L, datos.d, **arq, mu_y=datos.mu_y, sd_y=datos.sd_y).to(datos.dev)
    if estado is not None:
        m.load_state_dict(estado)
    return m.eval()


@torch.no_grad()
def pesos_anclas(m, datos, idx, y, bs=4096):
    W = np.concatenate([m.componentes(datos(b)[0])[2].float().cpu().numpy()
                        for b in np.array_split(idx, max(1, len(idx) // bs))])
    fuerte = y >= 6
    return dict(peso_medio=W.mean(0).tolist(), peso_medio_fuerte=W[fuerte].mean(0).tolist(),
                frecuencia_mayor_peso=np.bincount(W.argmax(1), minlength=W.shape[1]).astype(float)
                .__truediv__(len(W)).tolist(),
                entropia_media=float((-(W * np.log(W)).sum(1)).mean()), n=int(len(W)), n_fuerte=int(fuerte.sum()))


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(SALIDA, exist_ok=True)
    conf = json.load(open("resultados/e3/PADRE_configuracion.json"))
    hp, arq = conf["hp"], {**conf["arquitectura"], "w": tuple(conf["arquitectura"]["w"]),
                           "b": tuple(conf["arquitectura"]["b"])}
    cfg = leer_config()
    p = preparar_panel(cfg)
    f = pliegues(p, cfg, "prueba")[-1]
    assert f["origen"] == "2025-10-01"
    datos = Datos(p, f)
    ev, aj = f["evaluacion"], f["ajuste"]
    y_ev = datos.y[torch.as_tensor(ev + datos.h, device=datos.dev)].cpu().numpy()
    y_aj = datos.y[torch.as_tensor(aj + datos.h, device=datos.dev)].cpu().numpy()

    # 1) modelos de las 3 semillas
    modelos, reproduccion = {}, {}
    for s in SEMILLAS:
        ruta = f"{CONF}/{f['origen']}_{s}.pt" if s == 1000 else f"{SALIDA}/{f['origen']}_{s}.pt"
        if not os.path.exists(ruta):
            r = entrenar("PADRE", datos, f, s, hp=arq, lr=hp["lr"], wd=hp["wd"], bs=hp["bs"], epocas=50, paciencia=5)
            torch.save(dict(estado=r["estado"], hp=hp, arquitectura=arq, semilla=s, origen=f["origen"]), ruta)
            print(f"[{time.strftime('%H:%M')}] PADRE s{s} reentrenado ({r['segundos']:.0f} s)", flush=True)
        m = construir(datos, arq, torch.load(ruta, map_location=datos.dev, weights_only=False)["estado"])
        pred = predecir(m, datos, ev)
        guardada = np.load(f"{CONF}/{f['origen']}_{s}.npz")["pred"]
        reproduccion[s] = float(np.abs(pred - guardada).max())
        modelos[s] = m
    print("reproducción (máx |Δ| frente a las confirmatorias):", reproduccion, flush=True)

    # 2) gradientes integrados en las mismas 2 000 ventanas que el E2
    muestra = np.sort(np.random.default_rng(0).choice(ev, 2000, replace=False))
    perfiles_h, perfiles_v, fid, fuera_dentro, robustez, anclas = [], [], [], [], [], {}
    for s, m in modelos.items():
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
        # 3) pesos de las anclas
        anclas[s] = pesos_anclas(m, datos, ev, y_ev)
        # 4) robustez y razón fuera/dentro
        fuera = float(np.sqrt(np.mean((predecir(m, datos, ev) - y_ev) ** 2)))
        dentro = float(np.sqrt(np.mean((predecir(m, datos, aj) - y_aj) ** 2)))
        fuera_dentro.append(dict(semilla=s, rmse_prueba=fuera, rmse_ajuste=dentro, razon=fuera / dentro))
        robustez.append(dict(semilla=s, tipo="ninguna", nivel=0.0, rmse=fuera))
        for tipo, niveles in SEVERIDADES.items():
            for nivel in niveles:
                robustez.append(dict(semilla=s, tipo=tipo, nivel=nivel,
                                     rmse=rmse_perturbado(m, datos, ev, y_ev, tipo, nivel, semilla=s)))
        print(f"[{time.strftime('%H:%M')}] PADRE s{s}: fidelidad {np.mean(f_top):.3f} vs {np.mean(f_azar):.3f}; "
              f"fuera/dentro {fuera / dentro:.3f}", flush=True)
    from scipy.stats import spearmanr
    res = dict(origen=f["origen"], semillas=SEMILLAS, variables=list(p.cols), L=p.L, reproduccion=reproduccion,
               interpretabilidad=dict(
                   importancia_hora=np.mean(perfiles_h, 0).tolist(), importancia_variable=np.mean(perfiles_v, 0).tolist(),
                   importancia_hora_por_semilla=[x.tolist() for x in perfiles_h], fidelidad=fid,
                   spearman_horas=[float(spearmanr(perfiles_h[i], perfiles_h[j])[0])
                                   for i, j in itertools.combinations(range(3), 2)],
                   spearman_variables=[float(spearmanr(perfiles_v[i], perfiles_v[j])[0])
                                       for i, j in itertools.combinations(range(3), 2)]),
               anclas=anclas, fuera_dentro=fuera_dentro, robustez=robustez)
    json.dump(res, open(f"{SALIDA}/resultados.json", "w"), indent=1, ensure_ascii=False)
    print("Listo", flush=True)
