"""Infraestructura del Entregable 2: ventanas, partición con origen móvil y preprocesamiento por origen.

Pasos S1, S3 y S4 de la ruta de series de tiempo:

* S1. Frecuencia horaria, entrada de L horas, horizontes en `horizontes` (por defecto solo 24 h), estrategia
  directa (un modelo por horizonte) y ninguna covariable futura salvo el calendario, que es determinista.
* S3. Origen móvil: en cada origen T se entrena con lo anterior a T (ventana expansiva) y se evalúa con los
  pronósticos emitidos entre T y el origen siguiente. Entre el último objetivo de entrenamiento y el primer
  objetivo evaluado hay `embargo_horas` (igual al horizonte).
* S4. Imputación, escalado y rezagos solo con el pasado: el relleno hacia adelante mira atrás, el escalador se
  ajusta en cada origen con las filas de entrenamiento y los rezagos salen de la propia ventana de entrada.

Una ventana se identifica por la fila τ de su última hora de entrada (instante en que se emite el pronóstico):
entrada = filas τ-L+1 … τ, objetivos = velocidad en τ+h para cada h de `horizontes`.
"""
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

HORA = np.timedelta64(1, "h")


def leer_config(ruta="configs/e2.json"):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


@dataclass
class Panel:
    """Panel en una rejilla horaria regular, con las series de todas las estaciones una tras otra."""
    X: np.ndarray          # (N, d) float32 en unidades originales; NaN donde falta (tras el relleno causal)
    tiempo: np.ndarray     # (N,) datetime64[h]
    estacion: np.ndarray   # (N,) código de estación de cada fila
    tau: np.ndarray        # filas que pueden cerrar una ventana válida (entrada completa y objetivos observados)
    cols: list
    L: int
    horizontes: list

    @property
    def hmin(self):
        return min(self.horizontes)

    @property
    def hmax(self):
        return max(self.horizontes)

    def t(self, idx):
        """Instante de emisión del pronóstico de cada ventana."""
        return self.tiempo[idx]


def preparar_panel(cfg, df=None):
    """Construye el panel. `df` permite pasar datos ya cargados (se usa en las pruebas con datos sintéticos)."""
    cols, L, hs = cfg["columnas"], cfg["L"], sorted(cfg["horizontes"])
    if df is None:
        df = pd.read_csv(cfg["panel"], usecols=["CodigoEstacion", "hora"] + cols,
                         dtype={"CodigoEstacion": str}, parse_dates=["hora"])
    df = df[~df.CodigoEstacion.isin(cfg.get("excluir_estaciones", []))]
    X, tiempo, estacion, tau, ofs = [], [], [], [], 0
    for cod, g in df.groupby("CodigoEstacion", sort=True):
        g = g.set_index("hora").sort_index()
        g = g[~g.index.duplicated()].asfreq("h")                            # rejilla horaria regular
        x = g[cols].ffill(limit=cfg["ffill_max_horas"]).to_numpy(np.float32)  # el relleno solo mira atrás
        n = len(x)
        if n < L + hs[-1]:
            continue
        ok = ~np.isnan(x).any(1)
        c = np.concatenate([[0], np.cumsum(ok)])
        r = np.arange(L - 1, n - hs[-1])                                    # candidatos a τ dentro de la serie
        valida = c[r + 1] - c[r + 1 - L] == L                               # las L horas de entrada completas
        for h in hs:
            valida &= ~np.isnan(x[r + h, 0])                                # velocidad observada en τ + h
        X.append(x); tiempo.append(g.index.to_numpy().astype("datetime64[h]"))
        estacion.append(np.full(n, cod)); tau.append(ofs + r[valida]); ofs += n
    return Panel(np.concatenate(X), np.concatenate(tiempo), np.concatenate(estacion), np.concatenate(tau),
                 list(cols), L, hs)


# ---------------------------------------------------------------- S3: origen móvil

def _h(fecha):
    return np.datetime64(fecha, "h")


def dividir(panel, origen, fin, embargo, inicio=None):
    """Ventanas de entrenamiento y de evaluación para un origen.

    Evaluación: pronósticos emitidos en [origen, fin) cuyos objetivos también caen antes de `fin`.
    Entrenamiento: ventanas cuyo último objetivo es anterior al origen y está más de `embargo` horas antes del
    primer objetivo evaluable (origen + hmin); con `inicio` la ventana de entrenamiento es deslizante en lugar de
    expansiva. Así ningún dato del origen en adelante entra al entrenamiento.
    """
    T, Tn = _h(origen), _h(fin)
    t = panel.t(panel.tau)
    ultimo_obj = t + panel.hmax * HORA
    ev = (t >= T) & (ultimo_obj < Tn)
    tr = (ultimo_obj < T) & (ultimo_obj < T + (panel.hmin - embargo) * HORA)
    if inicio is not None:
        tr &= t - (panel.L - 1) * HORA >= _h(inicio)
    return panel.tau[tr], panel.tau[ev]


def pliegues(panel, cfg, etapa="validacion"):
    """Un pliegue por origen. Devuelve diccionarios con los índices τ de cada subconjunto.

    * `entrenamiento`: todo lo disponible antes del origen (para modelos sin parada temprana).
    * `ajuste` y `parada`: el entrenamiento partido en dos, con las últimas `parada_temprana_horas` como
      validación interna de la parada temprana y el mismo embargo entre ambas partes.
    * `evaluacion`: pronósticos emitidos entre este origen y el siguiente.
    """
    origenes = cfg["origenes_validacion"] if etapa == "validacion" else cfg["origenes_prueba"]
    # el último pliegue de validación termina donde empieza la prueba: ningún objetivo de 2025 entra a validación
    fines = origenes[1:] + [cfg["origenes_prueba"][0] if etapa == "validacion" else cfg["fin"]]
    emb = cfg["embargo_horas"]
    ini = None if cfg.get("ventana_entrenamiento", "expansiva") == "expansiva" else cfg["inicio"]
    salida = []
    for T, Tn in zip(origenes, fines):
        tr, ev = dividir(panel, T, Tn, emb, ini)
        S = str(_h(T) - cfg["parada_temprana_horas"] * HORA)
        aj, pa = dividir(panel, S, T, emb, ini)
        pa = pa[np.isin(pa, tr)]                         # la parada solo usa ventanas que ya eran de entrenamiento
        salida.append(dict(origen=T, fin=Tn, entrenamiento=tr, ajuste=aj, parada=pa, evaluacion=ev))
    return salida


# ---------------------------------------------------------------- S4: preprocesamiento por origen

def filas_de_entrada(panel, idx):
    """Filas del panel que alimentan las ventanas `idx` (sus L horas de entrada)."""
    return np.unique((idx[:, None] - np.arange(panel.L)).ravel())


class Escalador:
    """Estandarización con media y desviación de las filas de entrada del entrenamiento de un origen."""

    def ajustar(self, panel, idx_entrenamiento):
        x = panel.X[filas_de_entrada(panel, idx_entrenamiento)]
        self.mu, self.sd = np.nanmean(x, 0), np.nanstd(x, 0) + 1e-6
        return self

    def transformar(self, X):
        return np.nan_to_num((X - self.mu) / self.sd).astype(np.float32)


def entradas(X, idx, L):
    """Tensor (n, L, d) de ventanas de entrada a partir de una matriz ya escalada."""
    return X[idx[:, None] + np.arange(-L + 1, 1)]


def objetivos(panel, idx):
    """Velocidad observada (m/s) en τ + h: matriz (n, número de horizontes)."""
    return np.stack([panel.X[idx + h, 0] for h in panel.horizontes], 1)


def rezagos(panel, idx, lags_vel):
    """Variables tabulares para los modelos clásicos, calculadas solo con la ventana de entrada de cada τ.

    Incluye rezagos de velocidad, medias y desviación móviles de 24 y 168 h, el último valor de las demás
    columnas y el calendario del instante de emisión (hora del día y día del año en seno y coseno). El
    calendario es la única covariable futura: el del objetivo es el de τ desplazado h horas, que se conoce.
    """
    assert max(lags_vel) < panel.L, "un rezago no puede salir de la ventana de entrada"
    v = panel.X[:, 0]
    f = {f"vel_l{k}": v[idx - k] for k in lags_vel}
    for w in (24, panel.L):
        ventana = v[idx[:, None] - np.arange(w)]
        f[f"vel_media{w}"] = np.nanmean(ventana, 1)
        f[f"vel_std{w}"] = np.nanstd(ventana, 1)
    for j, c in enumerate(panel.cols[1:], 1):
        f[f"{c}_l0"] = panel.X[idx, j]
    t = pd.DatetimeIndex(panel.tiempo[idx])
    f["hora_sin"], f["hora_cos"] = np.sin(2 * np.pi * t.hour / 24), np.cos(2 * np.pi * t.hour / 24)
    f["dia_sin"], f["dia_cos"] = np.sin(2 * np.pi * t.dayofyear / 365.25), np.cos(2 * np.pi * t.dayofyear / 365.25)
    return pd.DataFrame(f)


# ---------------------------------------------------------------- resumen de los pliegues

if __name__ == "__main__":
    import sys, time
    cfg = leer_config(sys.argv[1] if len(sys.argv) > 1 else "configs/e2.json")
    t0 = time.time()
    p = preparar_panel(cfg)
    print(f"Panel: {len(p.X):,} filas, {len(np.unique(p.estacion))} estaciones, {len(p.tau):,} ventanas válidas "
          f"({time.time() - t0:.0f} s)")
    filas = []
    for etapa in ["validacion", "prueba"]:
        for f in pliegues(p, cfg, etapa):
            fila = dict(etapa=etapa, origen=f["origen"], fin=f["fin"])
            for k in ["entrenamiento", "ajuste", "parada", "evaluacion"]:
                fila[f"n_{k}"] = len(f[k])
            fila["estaciones_eval"] = len(np.unique(p.estacion[f["evaluacion"]]))
            if etapa == "validacion":                    # la prueba no se mira hasta después del prerregistro
                ev = f["evaluacion"]
                for h in p.horizontes:
                    fila[f"rmse_persistencia_h{h}"] = float(np.sqrt(np.mean((p.X[ev, 0] - p.X[ev + h, 0]) ** 2)))
            filas.append(fila)
    tabla = pd.DataFrame(filas)
    print(tabla.to_string(index=False))
    import os
    os.makedirs("resultados/e2", exist_ok=True)
    tabla.to_csv("resultados/e2/pliegues.csv", index=False)
