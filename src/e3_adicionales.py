"""Corridas en los conjuntos de datos adicionales del E3 (sección 8.3): NOAA ISD-Lite Kansas, SDWPF y ETTh1.

Mismo protocolo que las corridas finales del E2 (`finales.py`), con el recorte por cómputo declarado en
experiments/conjuntos_adicionales.md antes de entrenar:
  * 2 orígenes de prueba por conjunto (los de `configs/e3_{conjunto}.json`), reentrenando desde cero en cada uno;
  * cada modelo usa **la configuración elegida en el E2** (resultados/e2/finales/{modelo}_configuracion.json), sin
    reajustar en ningún conjunto; el modelo original usa la suya, fijada en el E3 (resultados/e3/PADRE_configuracion.json);
  * 10 semillas (1000–1009), 50 épocas como máximo, paciencia 5.
Baselines (`--baselines`, sin GPU, en ml_env y sin modelos en la misma llamada): persistencia, naive semanal, media de la misma hora en 7 días, ETS por serie, ridge
y gradient boosting, con los mismos rezagos del E2. Chronos y Moirai van aparte, sin entrenar (cero disparos), con
`preentrenados.py --config configs/e3_{conjunto}.json --etapa prueba --salida resultados/e3/adicionales/{conjunto}/cero`:
su versión con cabezal necesita los embeddings de cada conjunto y no cabe en el presupuesto (se declara).
STGCN usa el grafo de cada conjunto: k = min(8, series − 1) vecinas; en ETTh1, con una sola serie, un grafo de un nodo.
Salida: resultados/e3/adicionales/{conjunto}/{modelo}.csv (una fila por origen y semilla) y las predicciones por
ventana en {modelo}/{origen}_{semilla}.npz; los objetivos y la serie de cada ventana en objetivos_{origen}.npz.
Se reanuda donde quedó.
Uso: python src/e3_adicionales.py etth1 TCN PADRE ... [--semillas 1000-1009] [--hasta 06:00] [--baselines]
     python src/e3_adicionales.py todos --ligeros          (los modelos rápidos, en el PC de noche)
"""
import argparse, csv, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from origen_movil import leer_config, objetivos, pliegues, preparar_panel, rezagos  # noqa: E402

CONJUNTOS = ["etth1", "sdwpf", "noaa_kansas"]           # del más pequeño al más grande
LIGEROS = ["MLP", "DLinear", "KAN", "CNN", "TCN", "PADRE", "PatchTST"]
PESADOS = ["LSTM", "GRU", "Transformer", "Informer", "STGCN"]
FINALES_E2 = "resultados/e2/finales"


def ya_hechos(ruta, clave):        # copia de busqueda_arq (que importa torch); así los baselines corren en ml_env
    if not os.path.exists(ruta):
        return {}
    with open(ruta, newline="", encoding="utf-8") as f:
        return {r[clave]: r for r in csv.DictReader(f)}


def anotar(ruta, fila):
    nuevo = not os.path.exists(ruta)
    with open(ruta, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fila))
        if nuevo:
            w.writeheader()
        w.writerow(fila)


def configuracion(nombre):
    ruta = "resultados/e3/PADRE_configuracion.json" if nombre == "PADRE" else f"{FINALES_E2}/{nombre}_configuracion.json"
    if not os.path.exists(ruta):
        sys.exit(f"Falta {ruta}: la configuración de {nombre} se fija antes de correr los conjuntos adicionales")
    c = json.load(open(ruta))
    if nombre == "PADRE":
        arq = c["arquitectura"]
        return c["hp"], {**arq, "w": tuple(arq["w"]), "b": tuple(arq["b"])}
    if nombre == "DLinear":            # salida desnormalizada como en el paper (corrección declarada del E3)
        return c["hp"], dict(desnormalizar=True)
    from fase2 import arquitectura
    return c["hp"], (arquitectura(c["arquitectura"]["id"]) if c["arquitectura"] else None)


def baselines(conjunto, cfg, p, salida):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from baselines import ets_por_estacion, triviales
    ruta = f"{salida}/baselines.csv"
    hechos = ya_hechos(ruta, "clave")
    for f in pliegues(p, cfg, "prueba"):
        if all(f"{m}|{f['origen']}" in hechos for m in ["Persistencia", "Ridge", "Gradient boosting"]):
            continue
        tr, ev = f["entrenamiento"], f["evaluacion"]
        y_tr, y_ev = objetivos(p, tr)[:, 0], objetivos(p, ev)[:, 0]
        X_tr, X_ev = rezagos(p, tr, cfg["rezagos_vel"]), rezagos(p, ev, cfg["rezagos_vel"])
        mod = triviales(p, ev)
        mod["ETS por estación"], _ = ets_por_estacion(p, f["origen"], ev)
        mod["Ridge"] = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X_tr, y_tr).predict(X_ev)
        mod["Gradient boosting"] = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=63,
                                                                 early_stopping=False, random_state=0
                                                                 ).fit(X_tr, y_tr).predict(X_ev)
        for m, yhat in mod.items():
            yhat = np.asarray(yhat, dtype=np.float32)
            os.makedirs(f"{salida}/{m.replace(' ', '_')}", exist_ok=True)
            np.savez_compressed(f"{salida}/{m.replace(' ', '_')}/{f['origen']}.npz", tau=ev, pred=yhat)
            e = yhat - y_ev
            anotar(ruta, dict(clave=f"{m}|{f['origen']}", conjunto=conjunto, modelo=m, origen=f["origen"], n=len(ev),
                              rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e)))))
            print(f"[{time.strftime('%H:%M')}] {conjunto} {m} {f['origen']} → RMSE {np.sqrt(np.mean(e ** 2)):.4f}",
                  flush=True)


def correr(conjunto, modelos, semillas, hasta, con_baselines):
    cfg = leer_config(f"configs/e3_{conjunto}.json")
    salida = f"resultados/e3/adicionales/{conjunto}"
    os.makedirs(salida, exist_ok=True)
    conf = {m: configuracion(m) for m in modelos}
    t0 = time.time()
    p = preparar_panel(cfg)
    print(f"{conjunto}: {len(p.tau):,} ventanas válidas, {len(np.unique(p.estacion))} series "
          f"({time.time() - t0:.0f} s)", flush=True)
    if con_baselines:                  # baselines en ml_env (scikit-learn y statsmodels, sin GPU)
        baselines(conjunto, cfg, p, salida)
    if not modelos:
        return
    import torch                       # modelos de Deep Learning en dl_env
    from modelos_e2 import Datos, entrenar
    from busqueda_arq import registrar_hardware
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    for f in pliegues(p, cfg, "prueba"):
        ev = np.asarray(f["evaluacion"])
        y_ev = objetivos(p, ev)[:, 0]
        ruta_y = f"{salida}/objetivos_{f['origen']}.npz"
        if not os.path.exists(ruta_y):
            np.savez_compressed(ruta_y, tau=ev, y=y_ev, serie=p.estacion[ev], persistencia=p.X[ev, 0])
        pend = [(m, s) for m in modelos for s in semillas
                if f"{f['origen']}|{s}" not in ya_hechos(f"{salida}/{m}.csv", "clave")]
        if not pend:
            continue
        datos = Datos(p, f)
        if any(m == "STGCN" for m, _ in pend):
            datos.activar_grafo(p, cfg["panel"])
        for m, s in pend:
            if hasta and hasta <= time.strftime("%H:%M") < "18:00":
                print(f"[{time.strftime('%H:%M')}] Hora de parada ({hasta}): se retoma en la próxima corrida", flush=True)
                sys.exit(0)
            hp, arq = conf[m]
            registrar_hardware(m, f"E3 adicionales {conjunto}")
            r = entrenar(m, datos, f, s, hp=arq, lr=hp["lr"], wd=hp["wd"], bs=hp["bs"], epocas=50, paciencia=5)
            os.makedirs(f"{salida}/{m}", exist_ok=True)
            np.savez_compressed(f"{salida}/{m}/{f['origen']}_{s}.npz", tau=ev.astype(np.int32),
                                pred=r["pred"].astype(np.float32))
            err = r["pred"] - y_ev
            fila = dict(clave=f"{f['origen']}|{s}", conjunto=conjunto, modelo=m, origen=f["origen"], semilla=s,
                        n=len(ev), rmse=float(np.sqrt(np.mean(err ** 2))), mae=float(np.mean(np.abs(err))),
                        rmse_parada=r["rmse_parada"], epocas=r["epocas"], segundos=round(r["segundos"], 1),
                        parametros=r["parametros"])
            anotar(f"{salida}/{m}.csv", fila)
            print(f"[{time.strftime('%H:%M')}] {conjunto} {m} {f['origen']} s{s} → RMSE {fila['rmse']:.4f} "
                  f"({fila['segundos']} s)", flush=True)
        del datos
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("conjunto", choices=CONJUNTOS + ["todos"])
    ap.add_argument("modelos", nargs="*")
    ap.add_argument("--ligeros", action="store_true", help=f"modelos rápidos: {', '.join(LIGEROS)}")
    ap.add_argument("--pesados", action="store_true", help=f"modelos lentos (Colab): {', '.join(PESADOS)}")
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--semillas", default="1000-1009")
    ap.add_argument("--hasta", default=None, help="HH:MM: no empieza corridas nuevas después de esa hora (PC de noche)")
    a = ap.parse_args()
    modelos = list(a.modelos) + (LIGEROS if a.ligeros else []) + (PESADOS if a.pesados else [])
    if not modelos and not a.baselines:
        ap.error("indica modelos, --ligeros, --pesados o --baselines")
    s0, s1 = map(int, a.semillas.split("-"))
    for c in (CONJUNTOS if a.conjunto == "todos" else [a.conjunto]):
        correr(c, list(dict.fromkeys(modelos)), list(range(s0, s1 + 1)), a.hasta, a.baselines)
    print("Conjuntos adicionales: completo", flush=True)
