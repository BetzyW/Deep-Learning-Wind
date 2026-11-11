"""Búsqueda de arquitecturas MLP con timeseries-cv (sección 2.21 del curso), una estación, h = 24.

Qué hace, en el orden de la sección 2.21:
1. Toma las últimas `n_observaciones` horas de velocidad de la estación elegida (la de menos faltantes) y rellena
   hacia adelante solo los huecos de hasta 3 h (el relleno solo mira atrás).
2. Reserva el 10 % final como holdout (nunca menos de 2H). El holdout no entra en la validación cruzada, ni en la
   búsqueda, ni en los escaladores; se usa solo como evidencia final.
3. Parte el resto con `tsxv.splitTrainVal.split_train_val_forwardChaining(serie, I, 24, salto)`. Para no copiar
   ≈ 90 000 ventanas por pliegue, la función se aplica a la serie de ÍNDICES 0..n-1: devuelve exactamente qué
   ventanas usa cada pliegue y con esos índices se leen los valores. Las ventanas que tocan un hueco de más de 3 h se
   descartan (la librería no admite faltantes).
4. Rejilla: 1 a 3 capas con {32, 64, 128, 256} neuronas, Dropout(0.5) activable después de cada capa (2^capas
   máscaras): 584 configuraciones por entrada; entradas 24, 48, 96 y 192 h.
5. Cada configuración se entrena en CADA pliegue como en Keras: Dense ReLU (+Dropout) y salida Dense(24) lineal,
   inicialización Glorot uniforme, Adam (lr 0.001, eps 1e-7), pérdida MSE, 30 épocas, lote 32, sin barajar, y
   StandardScaler de X e y ajustado con el entrenamiento del pliegue.
6. Selección: gap_ratio = |RMSE de entrenamiento final − mejor RMSE de validación| / mejor RMSE de validación;
   se filtra gap_ratio < 0.15 y se ordena por el mejor RMSE de validación.

Cómo se acelera sin cambiar el método: en forward chaining el pliegue k entrena con las primeras k+2 ventanas de una
misma lista, así que todos los pliegues comparten el orden de los lotes. Los ≈ 430 modelos de una configuración (uno
por pliegue, cada uno con sus propios pesos, escaladores, Dropout y estado de Adam) se entrenan a la vez como un
tensor por lotes en la GPU. En el paso s solo se actualizan los pliegues que tienen un lote s; los demás no cambian
ni sus pesos ni su Adam, igual que si se entrenaran por separado.

Curvas por época (como el history de Keras, en la escala estandarizada): RMSE de entrenamiento acumulado en la época
y RMSE de validación con la red en modo evaluación; ambas se promedian entre pliegues. Se reporta además el RMSE de
validación en m/s.

Uso:
  python src/timeseries_cv.py                       # corre todo (se puede retomar: salta lo ya guardado)
  python src/timeseries_cv.py --hasta 06:00         # se detiene limpio antes de esa hora
  python src/timeseries_cv.py --piloto 6            # mide el tiempo con 6 configuraciones por entrada y proyecta
  python src/timeseries_cv.py --resumen             # tabla de seleccionados (gap_ratio < 0.15) por entrada
"""
import argparse
import datetime as dt
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from tsxv.splitTrainVal import split_train_val_forwardChaining


# ---------------------------------------------------------------- datos y pliegues


def leer_serie(cfg):
    df = pd.read_csv(cfg["panel"], usecols=["CodigoEstacion", "hora", "vel"], dtype={"CodigoEstacion": str},
                     parse_dates=["hora"])
    fin = df.hora.max()
    s = df[df.CodigoEstacion == cfg["estacion"]].set_index("hora").vel.sort_index()
    s = s[~s.index.duplicated()]
    idx = pd.date_range(end=fin, periods=cfg["n_observaciones"], freq="h")
    s = s.reindex(idx).ffill(limit=cfg["ffill_max_horas"])
    return s


def pliegues_tsxv(n_cv, I, O, J):
    """Aplica tsxv a la serie de índices y devuelve, por pliegue, el inicio de sus ventanas de entrenamiento y el
    inicio de su ventana de validación."""
    X, _, Xcv, _ = split_train_val_forwardChaining(np.arange(n_cv), I, O, J)
    inicios_tr = [X[k][:, 0] for k in range(len(X))]
    inicios_cv = np.array([Xcv[k][0, 0] for k in range(len(X))])
    # comprobación: el entrenamiento del pliegue k son las primeras k+2 ventanas de la lista 0, J, 2J, ...
    for k, a in enumerate(inicios_tr):
        assert np.array_equal(a, J * np.arange(k + 2)), "tsxv cambió su esquema de pliegues"
    return inicios_tr, inicios_cv


class Lotes:
    """Datos de entrenamiento compartidos por F modelos (uno por pliegue): el modelo f entrena con las primeras m[f]
    filas de (X, Y) y valida con sus V ventanas Xcv[f], Ycv[f]. Escaladores propios de cada modelo."""

    def __init__(self, X, Y, m, Xcv, Ycv, dev):
        def escalar(A):                       # StandardScaler (ddof = 0) del prefijo de m filas, por columna
            s1 = np.cumsum(A.astype(np.float64), 0)
            s2 = np.cumsum(A.astype(np.float64) ** 2, 0)
            mu = s1[m - 1] / m[:, None]
            sd = np.sqrt(np.maximum(s2[m - 1] / m[:, None] - mu ** 2, 0))
            sd[sd < 1e-8] = 1.0
            return mu.astype(np.float32), sd.astype(np.float32)
        mx, sx = escalar(X)
        my, sy = escalar(Y)
        t = lambda a: torch.tensor(np.asarray(a), device=dev)                    # noqa: E731
        self.m = m
        self.Xcv_crudo, self.Ycv_crudo = Xcv, Ycv
        self.Xg, self.Yg, self.mg = t(X.astype(np.float32)), t(Y.astype(np.float32)), t(m)
        self.mx, self.sx, self.my, self.sy = t(mx)[:, None], t(sx)[:, None], t(my)[:, None], t(sy)[:, None]
        self.Xcv = (t(Xcv.astype(np.float32)) - self.mx) / self.sx              # (F, V, I)
        self.Ycv_ms = t(Ycv.astype(np.float32))                                  # (F, V, O)
        self.Ycv = (self.Ycv_ms - self.my) / self.sy
        self.F, self.I, self.O = len(m), X.shape[1], Y.shape[1]


def ventanas(v, inicios, I, O):
    W = np.stack([v[x:x + I + O] for x in inicios])
    return W[:, :I], W[:, I:]


def completas(v, I, O):
    c = np.concatenate([[0], np.cumsum(~np.isnan(v))])
    return lambda a: c[np.asarray(a) + I + O] - c[np.asarray(a)] == I + O


def problema_cv(v, n_cv, I, O, J, dev):
    """Pliegues de tsxv para la entrada I, sin las ventanas que tocan huecos de más de 3 h."""
    completa = completas(v, I, O)
    inicios_tr, inicios_cv = pliegues_tsxv(n_cv, I, O, J)
    todas = J * np.arange(len(inicios_tr) + 1)                                  # ventanas que usa el último pliegue
    validas = todas[completa(todas)]                                            # lista global, en orden temporal
    m = np.array([int(completa(a).sum()) for a in inicios_tr])                  # ventanas válidas de cada pliegue
    usar = (m >= 2) & completa(inicios_cv)
    X, Y = ventanas(v, validas, I, O)
    Xcv, Ycv = ventanas(v, inicios_cv[usar], I, O)
    P = Lotes(X, Y, m[usar], Xcv[:, None], Ycv[:, None], dev)
    P.n_pliegues_tsxv, P.descartados = len(inicios_tr), int((~usar).sum())
    return P


def problema_holdout(v, n_cv, I, O, J, dev):
    """Modelo final: entrena con todas las ventanas válidas del tramo de validación cruzada (paso J) y se evalúa en
    las ventanas del holdout (paso J; su entrada puede venir del final del tramo, nunca su objetivo)."""
    completa = completas(v, I, O)
    tr = J * np.arange((n_cv - I - O) // J + 1)
    tr = tr[completa(tr)]
    pr = np.arange(n_cv - I, len(v) - I - O + 1, J)
    pr = pr[completa(pr)]
    X, Y = ventanas(v, tr, I, O)
    Xp, Yp = ventanas(v, pr, I, O)
    return Lotes(X, Y, np.array([len(tr)]), Xp[None], Yp[None], dev)


# ---------------------------------------------------------------- modelos por lotes (un modelo por pliegue)


def glorot(F, a, b, gen, dev):
    lim = np.sqrt(6.0 / (a + b))
    return (torch.rand(F, a, b, generator=gen, device=dev) * 2 - 1) * lim


class AdamPorPliegue:
    """Adam de Keras (beta1 0.9, beta2 0.999, eps 1e-7) con contador de pasos propio de cada pliegue; un pliegue
    inactivo en el paso no cambia ni sus pesos ni sus momentos."""

    def __init__(self, params, lr, F, dev):
        self.p, self.lr = params, lr
        self.m1 = [torch.zeros_like(p) for p in params]
        self.m2 = [torch.zeros_like(p) for p in params]
        self.t = torch.zeros(F, device=dev)

    @torch.no_grad()
    def paso(self, activo):
        self.t += activo.float()
        b1, b2, eps = 0.9, 0.999, 1e-7
        tt = self.t.clamp(min=1)
        c1 = (1 - b1 ** tt)
        c2 = (1 - b2 ** tt)
        for p, m1, m2 in zip(self.p, self.m1, self.m2):
            g = p.grad
            forma = (-1,) + (1,) * (p.dim() - 1)
            a = activo.view(forma)
            m1.copy_(torch.where(a, b1 * m1 + (1 - b1) * g, m1))
            m2.copy_(torch.where(a, b2 * m2 + (1 - b2) * g * g, m2))
            # forma de Keras: lr_t = lr·sqrt(1-b2^t)/(1-b1^t);  p -= lr_t·m1/(sqrt(m2)+eps)
            lr_t = (self.lr * torch.sqrt(c2) / c1).view(forma)
            p.sub_(torch.where(a, lr_t * m1 / (m2.sqrt() + eps), torch.zeros_like(p)))
            p.grad = None


def entrenar_configuracion(P, capas, mascara, cfg, semilla, dev, iniciales=None):
    """Entrena la configuración en los F modelos a la vez. Devuelve las curvas por época (promedio entre modelos,
    escala estandarizada) y el RMSE de validación en m/s de cada modelo en cada época (épocas × F)."""
    gen = torch.Generator(device=dev).manual_seed(semilla)
    F, I, O = P.F, P.I, P.O
    dims = [I] + list(capas) + [O]
    if iniciales is None:
        iniciales = [glorot(F, a, b, gen, dev) for a, b in zip(dims[:-1], dims[1:])]
    W = [w.clone().requires_grad_(True) for w in iniciales]
    B = [torch.zeros(F, 1, b, device=dev, requires_grad=True) for b in dims[1:]]
    opt = AdamPorPliegue(W + B, cfg["tasa_aprendizaje"], F, dev)
    q = cfg["dropout"]

    def red(x, entrenando):
        h = x
        for li in range(len(capas)):
            h = torch.relu(torch.baddbmm(B[li], h, W[li]))
            if mascara[li] and entrenando:
                h = h * (torch.rand(h.shape, generator=gen, device=dev) >= q) / (1 - q)
        return torch.baddbmm(B[-1], h, W[-1])

    lote, M = cfg["lote"], int(P.m.max())
    pasos = (M + lote - 1) // lote
    curva_tr, curva_cv, cv_ms = [], [], []
    for _ in range(cfg["epocas"]):
        sse = torch.zeros(F, device=dev)
        for s in range(pasos):
            r0, r1 = s * lote, min(s * lote + lote, M)
            activo = P.mg > r0
            filas = torch.arange(r0, r1, device=dev)
            mask = (filas[None, :] < P.mg[:, None]).float()                       # (F, b)
            x = (P.Xg[r0:r1][None] - P.mx) / P.sx                                # (F, b, I)
            y = (P.Yg[r0:r1][None] - P.my) / P.sy
            err = ((red(x, True) - y) ** 2).mean(-1)                             # (F, b)
            n = mask.sum(1).clamp(min=1)
            perdida = ((err * mask).sum(1) / n).sum()
            perdida.backward()
            opt.paso(activo)
            sse += (err.detach() * mask).sum(1)
        curva_tr.append(torch.sqrt(sse / P.mg).mean().item())
        with torch.no_grad():
            pred = red(P.Xcv, False)                                             # (F, V, O)
            rmse = torch.sqrt(((pred - P.Ycv) ** 2).mean((1, 2)))
            rmse_ms = torch.sqrt(((pred * P.sy + P.my - P.Ycv_ms) ** 2).mean((1, 2)))
        curva_cv.append(rmse.mean().item())
        cv_ms.append(rmse_ms.cpu().numpy())
    return np.array(curva_tr), np.array(curva_cv), np.stack(cv_ms)               # cv_ms: (épocas, F)


# ---------------------------------------------------------------- rejilla y ejecución


def rejilla(cfg):
    for capas_n in range(1, cfg["max_capas"] + 1):
        for capas in itertools.product(cfg["neuronas"], repeat=capas_n):
            for mascara in itertools.product([0, 1], repeat=capas_n):
                yield capas, mascara


def ident(capas, mascara):
    return "w" + "-".join(map(str, capas)) + "_d" + "".join(map(str, mascara))


def parametros(I, O, capas):
    d = [I] + list(capas) + [O]
    return int(sum(a * b + b for a, b in zip(d[:-1], d[1:])))


def principal():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tsxv.json")
    ap.add_argument("--hasta", default=None, help="HH:MM: se detiene limpio al pasar esa hora")
    ap.add_argument("--piloto", type=int, default=0, help="n configuraciones por entrada para medir el tiempo")
    ap.add_argument("--entradas", type=int, nargs="*", default=None)
    ap.add_argument("--resumen", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    cfg = json.load(open(a.config, encoding="utf-8"))
    os.makedirs(cfg["salida"], exist_ok=True)
    archivo = os.path.join(cfg["salida"], "busqueda.csv")
    if a.resumen:
        return resumen(cfg, archivo)

    dev = "cpu" if a.cpu or not torch.cuda.is_available() else "cuda"
    limite = None
    if a.hasta:
        h, mi = map(int, a.hasta.split(":"))
        ahora = dt.datetime.now()
        limite = ahora.replace(hour=h, minute=mi, second=0, microsecond=0)
        if limite <= ahora:
            limite += dt.timedelta(days=1)

    s = leer_serie(cfg)
    v = s.to_numpy(np.float64)
    O, J = cfg["horizonte"], cfg["salto"]
    n_hold = max(int(round(len(v) * cfg["holdout_fraccion"])), 2 * O)
    n_cv = len(v) - n_hold
    print(f"Estación {cfg['estacion']} · {s.index[0]} → {s.index[-1]} · {len(v):,} h · faltantes tras relleno "
          f"{int(np.isnan(v).sum())} · validación cruzada {n_cv:,} h · holdout {n_hold:,} h "
          f"(desde {s.index[n_cv]}) · dispositivo {dev}", flush=True)
    meta = dict(estacion=cfg["estacion"], desde=str(s.index[0]), hasta=str(s.index[-1]), n=len(v), n_cv=n_cv,
                n_holdout=n_hold, faltantes=int(np.isnan(v).sum()), dispositivo=dev, pliegues={})

    hechos = set()
    if os.path.exists(archivo) and not a.piloto:
        hechos = set(pd.read_csv(archivo, usecols=["clave"]).clave)
    configs = list(rejilla(cfg))
    entradas = a.entradas or cfg["entradas"]
    tiempos = {}
    for I in entradas:
        P = problema_cv(v, n_cv, I, O, J, dev)
        H = problema_holdout(v, n_cv, I, O, J, dev)
        meta["pliegues"][I] = dict(tsxv=P.n_pliegues_tsxv, usados=P.F, descartados_por_faltantes=P.descartados,
                                   ventanas_entrenamiento_max=int(P.m.max()))
        print(f"I={I}: tsxv generó {P.n_pliegues_tsxv} pliegues; se usan {P.F} ({P.descartados} tocan huecos > 3 h)"
              f"; hasta {int(P.m.max())} ventanas de entrenamiento", flush=True)
        lista = configs[:a.piloto] if a.piloto else configs
        if a.piloto:                                  # piloto: configuraciones de los tres tamaños de red
            lista = [configs[0], configs[len(configs) // 2], configs[-1]] + configs[1:max(1, a.piloto - 2)]
            lista = lista[:a.piloto]
        t_ent = []
        for j, (capas, mascara) in enumerate(lista):
            clave = f"I{I}|{ident(capas, mascara)}"
            if clave in hechos:
                continue
            if limite and dt.datetime.now() >= limite:
                print(f"Hora límite {a.hasta}: se detiene; se retoma donde quedó.", flush=True)
                guardar_meta(cfg, meta)
                return
            semilla = cfg["semilla_base"] + configs.index((capas, mascara))
            t0 = time.time()
            tr, cv, cv_ms = entrenar_configuracion(P, capas, mascara, cfg, semilla, dev)
            _, _, hold = entrenar_configuracion(H, capas, mascara, cfg, semilla + 100000, dev)
            hold, n_hp = float(hold[-1, 0]), H.Xcv.shape[1]
            seg = time.time() - t0
            t_ent.append(seg)
            e = int(np.argmin(cv))
            gap = abs(tr[-1] - cv[e])
            fila = dict(clave=clave, entrada=I, arquitectura=ident(capas, mascara), capas=len(capas),
                        neuronas="-".join(map(str, capas)), mascara_dropout="".join(map(str, mascara)),
                        parametros=parametros(I, O, capas), pliegues=P.F, best_val_rmse=cv[e], epoca_best=e + 1,
                        last_train_rmse=tr[-1], gap=gap, gap_ratio=gap / cv[e],
                        best_val_rmse_ms=float(cv_ms[e].mean()), best_val_rmse_ms_sd=float(cv_ms[e].std()),
                        holdout_rmse_ms=hold, ventanas_holdout=n_hp, segundos=round(seg, 2),
                        train_curve=json.dumps([round(x, 5) for x in tr]),
                        val_curve=json.dumps([round(x, 5) for x in cv]),
                        val_rmse_ms_pliegues=json.dumps([round(float(x), 4) for x in cv_ms[e]]))
            if not a.piloto:
                pd.DataFrame([fila]).to_csv(archivo, mode="a", header=not os.path.exists(archivo), index=False)
            if a.piloto or j % 25 == 0:
                print(f"  {clave}: val {cv[e]:.4f} (época {e + 1}) · gap_ratio {gap / cv[e]:.3f} · "
                      f"holdout {hold:.3f} m/s · {seg:.1f} s", flush=True)
        tiempos[I] = t_ent
    guardar_meta(cfg, meta)
    if a.piloto:
        total = sum(np.mean(t) * len(configs) for t in tiempos.values() if t)
        print(f"\nPiloto: {', '.join(f'I={I}: {np.mean(t):.1f} s/config' for I, t in tiempos.items() if t)}")
        print(f"Proyección de la búsqueda completa ({len(configs)} × {len(tiempos)} entradas): "
              f"≈ {total / 3600:.1f} h en {dev}")


def guardar_meta(cfg, meta):
    with open(os.path.join(cfg["salida"], "datos_muestra.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)


def resumen(cfg, archivo):
    d = pd.read_csv(archivo)
    print(f"{len(d)} configuraciones evaluadas")
    for I, g in d.groupby("entrada"):
        c = g[g.gap_ratio < cfg["gap_ratio_max"]].sort_values("best_val_rmse")
        print(f"\nEntrada {I} h: {len(c)} de {len(g)} pasan gap_ratio < {cfg['gap_ratio_max']}")
        print(c.head(5)[["arquitectura", "parametros", "best_val_rmse", "best_val_rmse_ms", "gap_ratio",
                         "holdout_rmse_ms"]].to_string(index=False))


if __name__ == "__main__":
    sys.exit(principal())
