"""Convierte los tres conjuntos adicionales del E3 al formato de panel del proyecto y escribe su configuración.

Formato de panel (el mismo de `origen_movil.preparar_panel`): una fila por serie y hora, con `CodigoEstacion`, `hora`,
`lat`, `lon` y las columnas de entrada, la primera de las cuales es el objetivo. Así los 14 modelos y el modelo original
corren sobre los tres conjuntos con el mismo código, la misma ventana (168 h), el mismo horizonte (24 h), el mismo
embargo y la misma lógica de origen móvil.
* NOAA ISD-Lite, Kansas (2019–2024): columnas vel, dir_sin, dir_cos, temp; hora en UTC. Formato de ancho fijo de NOAA
  (año, mes, día, hora, temperatura ×10, rocío, presión, dirección, velocidad ×10, …; −9999 = faltante).
* SDWPF: objetivo potencia activa (kW, horaria) y entradas velocidad, dirección absoluta (Ndir + Wdir), temperatura
  exterior. El conjunto solo trae el número de día, así que la hora es relativa (día 1 = 2000-01-01, declarado). Se
  limpian los registros anómalos que define el reto (Patv ≤ 0 con Wspd > 2.5 m/s, o algún ángulo de paso > 89°) y la
  potencia negativa se lleva a 0. Las posiciones (x, y) en metros se pasan a grados con 1° ≈ 111 320 m, solo para que
  STGCN arme su grafo por distancia.
* ETTh1: objetivo OT (temperatura del aceite) y las 6 cargas; una sola serie sin coordenadas.
Recorte por cómputo (decidido antes de entrenar ningún modelo en estos conjuntos y aplicado igual a todos): 2 orígenes
de prueba por conjunto en lugar de 4, y en NOAA 10 de las 34 estaciones (al azar, semilla 11) y el periodo 2021–2024.
Salida: datos/e3/{conjunto}/panel.csv y configs/e3_{conjunto}.json
Uso: python -I src/e3_conjuntos.py      (ml_env; lee solo datos/e3/)
"""
import glob, gzip, json, os
import numpy as np
import pandas as pd

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(RAIZ, "datos", "e3")
BASE = dict(ffill_max_horas=3, frecuencia="1h", L=168, horizontes=[24], estrategia_multipaso="directa",
            covariables_futuras="ninguna", ventana_entrenamiento="expansiva", embargo_horas=24,
            reentrenar_en_cada_origen=True, excluir_estaciones=[], contrato=False,
            rezagos_vel=[0, 1, 2, 3, 6, 12, 18, 23, 24, 47, 48, 71, 72, 95, 96, 119, 120, 143, 144, 167])


N_NOAA = 10          # recorte por cómputo, declarado antes de entrenar: 10 de las 34 estaciones al azar (semilla 11)


def noaa():
    est = pd.read_csv(os.path.join(D, "noaa_kansas", "estaciones_candidatas.csv"), dtype=str)
    est = est.iloc[np.sort(np.random.default_rng(11).choice(len(est), N_NOAA, replace=False))]
    filas = []
    for r in est.itertuples():
        cod = f"{r.USAF}{r.WBAN}"
        for ruta in sorted(glob.glob(os.path.join(D, "noaa_kansas", "isd_lite", f"{r.USAF}-{r.WBAN}-*.gz"))):
            with gzip.open(ruta, "rt") as fh:
                t = pd.read_fwf(fh, widths=[4, 3, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6], header=None,
                                names=["a", "m", "d", "h", "temp", "rocio", "slp", "dir", "vel", "cielo", "p1", "p6"])
            t = t.replace(-9999, np.nan)
            hora = pd.to_datetime(dict(year=t.a, month=t.m, day=t.d, hour=t.h))
            vel = t.vel / 10.0
            rad = np.deg2rad(t["dir"])
            calma = vel == 0
            filas.append(pd.DataFrame(dict(CodigoEstacion=cod, hora=hora, lat=float(r.LAT), lon=float(r.LON),
                                           vel=vel, dir_sin=np.where(calma, 0.0, np.sin(rad)),
                                           dir_cos=np.where(calma, 0.0, np.cos(rad)), temp=t.temp / 10.0)))
    p = pd.concat(filas).drop_duplicates(["CodigoEstacion", "hora"])
    p = p[p.hora >= "2021-01-01"]                                         # recorte por cómputo: 2021–2024
    cfg = dict(BASE, columnas=["vel", "dir_sin", "dir_cos", "temp"], inicio="2021-01-01", fin="2025-01-01",
               parada_temprana_horas=2160,
               origenes_validacion=["2024-01-01", "2024-04-01"],
               origenes_prueba=["2024-07-01", "2024-10-01"])
    return p, cfg


def sdwpf():
    s = pd.read_csv(os.path.join(D, "sdwpf", "sdwpf_245days_v1.csv"))
    ub = pd.read_csv(os.path.join(D, "sdwpf", "sdwpf_baidukddcup2022_turb_location.csv"), encoding="utf-8-sig")
    anomala = ((s.Patv <= 0) & (s.Wspd > 2.5)) | (s[["Pab1", "Pab2", "Pab3"]] > 89).any(axis=1)
    s.loc[anomala, ["Patv"]] = np.nan
    s["Patv"] = s.Patv.clip(lower=0)
    hh, mm = s.Tmstamp.str.split(":", expand=True).astype(int).T.values
    s["hora"] = (pd.Timestamp("2000-01-01") + pd.to_timedelta(s.Day - 1, "D") + pd.to_timedelta(hh, "h")).values
    rad = np.deg2rad(s.Ndir + s.Wdir)
    s["dir_sin"], s["dir_cos"] = np.sin(rad), np.cos(rad)
    g = s.groupby(["TurbID", "hora"])[["Patv", "Wspd", "dir_sin", "dir_cos", "Etmp"]].mean().reset_index()
    g = g.merge(ub, on="TurbID")
    p = pd.DataFrame(dict(CodigoEstacion=g.TurbID.map(lambda i: f"T{i:03d}"), hora=g.hora, lat=g.y / 111320.0,
                          lon=g.x / 111320.0, potencia=g.Patv, vel=g.Wspd, dir_sin=g.dir_sin, dir_cos=g.dir_cos,
                          temp=g.Etmp))
    dia = lambda n: str((pd.Timestamp("2000-01-01") + pd.Timedelta(days=n - 1)).date())
    cfg = dict(BASE, columnas=["potencia", "vel", "dir_sin", "dir_cos", "temp"], inicio=dia(1), fin=dia(246),
               parada_temprana_horas=336,
               origenes_validacion=[dia(n) for n in (185, 200)],
               origenes_prueba=[dia(n) for n in (215, 230)])
    return p, cfg


def etth1():
    e = pd.read_csv(os.path.join(D, "etth1", "ETTh1.csv"), parse_dates=["date"])
    cols = ["OT", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
    p = pd.DataFrame(dict(CodigoEstacion="ETTh1", hora=e.date, lat=0.0, lon=0.0, **{c: e[c] for c in cols}))
    cfg = dict(BASE, columnas=cols, inicio="2016-07-01", fin="2018-03-01", parada_temprana_horas=720,
               origenes_validacion=["2017-11-01", "2017-12-01"],
               origenes_prueba=["2018-01-01", "2018-02-01"])
    return p, cfg


if __name__ == "__main__":
    for nombre, f in [("noaa_kansas", noaa), ("sdwpf", sdwpf), ("etth1", etth1)]:
        p, cfg = f()
        ruta = os.path.join(D, nombre, "panel.csv")
        p.to_csv(ruta, index=False)
        cfg["panel"] = f"datos/e3/{nombre}/panel.csv"
        json.dump(cfg, open(os.path.join(RAIZ, "configs", f"e3_{nombre}.json"), "w", encoding="utf-8"), indent=1)
        print(nombre, p.shape, p.CodigoEstacion.nunique(), "series;", p.hora.min(), "->", p.hora.max(),
              "| faltantes del objetivo:", round(float(p[cfg["columnas"][0]].isna().mean()), 3), flush=True)
