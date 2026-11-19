"""Contrato de datos del panel (sección 10.2 del enunciado).

* ESQUEMA: tipo, unidad, rango permitido y fuente de cada columna que usa el E2, y si es causalmente anterior al
  objetivo (todas lo son: se observan hasta τ y el objetivo es la velocidad en τ + 24 h).
* validar(df): reglas que se ejecutan automáticamente cada vez que `origen_movil.preparar_panel` lee el panel; si una
  falla, se detiene con un error (no se entrena con datos que violen el contrato).
* Con `python src/contrato_datos.py` se escribe además el registro de poblaciones: cuántas filas y ventanas quedan
  después de cada paso, desde el archivo crudo hasta las ventanas de validación y de prueba.
Salida: resultados/e2/contrato_datos.json
"""
import json, os, sys
import numpy as np
import pandas as pd

ESQUEMA = {
    "CodigoEstacion": dict(tipo="texto de 10 dígitos", unidad="—", rango="código IDEAM", fuente="IDEAM"),
    "hora": dict(tipo="fecha-hora (UTC−5)", unidad="h", rango="2020-01-01 00:00 a 2025-12-31 23:00, en punto",
                 fuente="IDEAM, agregada a la hora"),
    "vel": dict(tipo="real", unidad="m/s", rango=(0.0, 40.0), fuente="IDEAM: velocidad del viento (media horaria)"),
    "dir_sin": dict(tipo="real", unidad="—", rango=(-1.0, 1.0), fuente="IDEAM: dirección, media vectorial horaria"),
    "dir_cos": dict(tipo="real", unidad="—", rango=(-1.0, 1.0), fuente="IDEAM: dirección, media vectorial horaria"),
    "temp": dict(tipo="real", unidad="°C", rango=(-5.0, 50.0), fuente="IDEAM: temperatura del aire (media horaria)"),
    "lat": dict(tipo="real", unidad="grados", rango=(7.0, 13.0), fuente="IDEAM: catálogo de estaciones"),
    "lon": dict(tipo="real", unidad="grados", rango=(-77.0, -71.0), fuente="IDEAM: catálogo de estaciones"),
}
OBLIGATORIAS = ["CodigoEstacion", "hora", "vel"]       # sin faltantes; dirección y temperatura pueden faltar


class ContratoRoto(ValueError):
    pass


def validar(df):
    """Comprueba el contrato sobre el panel crudo. Devuelve el número de filas revisadas."""
    errores = []
    faltan = [c for c in ESQUEMA if c in ("CodigoEstacion", "hora", "vel") and c not in df.columns]
    if faltan:
        raise ContratoRoto(f"faltan columnas {faltan}")
    if not pd.api.types.is_datetime64_any_dtype(df.hora):
        errores.append("hora no es fecha-hora")
    elif ((df.hora.dt.minute != 0) | (df.hora.dt.second != 0)).any():
        errores.append("hay horas que no están en punto")
    if not df.CodigoEstacion.astype(str).str.fullmatch(r"\d{10}").all():
        errores.append("códigos de estación que no tienen 10 dígitos")
    for c in OBLIGATORIAS:
        if df[c].isna().any():
            errores.append(f"{c} tiene faltantes")
    for c, e in ESQUEMA.items():
        if c in df.columns and isinstance(e["rango"], tuple):
            x = df[c].dropna()
            fuera = int(((x < e["rango"][0]) | (x > e["rango"][1])).sum())
            if fuera:
                errores.append(f"{c}: {fuera} valores fuera de {e['rango']}")
    if {"dir_sin", "dir_cos"} <= set(df.columns):
        modulo = df.dir_sin ** 2 + df.dir_cos ** 2
        if (modulo > 1 + 1e-5).any():
            errores.append("la media vectorial de la dirección tiene módulo mayor que 1")
        if (df.dir_sin.isna() != df.dir_cos.isna()).any():
            errores.append("seno y coseno de la dirección no faltan a la vez")
    if errores:
        raise ContratoRoto("; ".join(errores))
    return len(df)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    from origen_movil import leer_config, pliegues, preparar_panel
    cfg = leer_config()
    df = pd.read_csv(cfg["panel"], usecols=["CodigoEstacion", "hora", "lat", "lon"] + cfg["columnas"],
                     dtype={"CodigoEstacion": str}, parse_dates=["hora"])
    pasos = [dict(paso="Archivo del panel (salida del proyecto de ML)", filas=len(df),
                  estaciones=int(df.CodigoEstacion.nunique()))]
    validar(df)
    pasos.append(dict(paso="Contrato validado (tipos, rangos y obligatorias)", filas=len(df),
                      estaciones=int(df.CodigoEstacion.nunique())))
    df = df[~df.CodigoEstacion.isin(cfg["excluir_estaciones"])]
    pasos.append(dict(paso=f"Sin la estación {', '.join(cfg['excluir_estaciones'])} (anemómetro bloqueado)",
                      filas=len(df), estaciones=int(df.CodigoEstacion.nunique())))
    rej, relleno = 0, 0
    for _, g in df.groupby("CodigoEstacion"):
        g = g.set_index("hora").sort_index()
        g = g[~g.index.duplicated()].asfreq("h")
        rej += len(g)
        relleno += int(g[cfg["columnas"]].isna().any(axis=1).sum()
                       - g[cfg["columnas"]].ffill(limit=cfg["ffill_max_horas"]).isna().any(axis=1).sum())
    pasos.append(dict(paso="Rejilla horaria regular por estación (las horas sin registro quedan como faltantes)",
                      filas=rej, estaciones=int(df.CodigoEstacion.nunique())))
    pasos.append(dict(paso=f"Relleno hacia adelante de máximo {cfg['ffill_max_horas']} h (solo mira el pasado)",
                      filas=rej, filas_completadas=relleno))
    p = preparar_panel(cfg)
    pasos.append(dict(paso=f"Ventanas válidas: {p.L} h de entrada completas y objetivo observado a 24 h",
                      ventanas=int(len(p.tau)), estaciones=int(len(set(p.estacion[p.tau])))))
    for etapa in ("validacion", "prueba"):
        for f in pliegues(p, cfg, etapa):
            pasos.append(dict(paso=f"{dict(validacion='Validación', prueba='Prueba')[etapa]} {f['origen']}: entrenamiento / evaluación",
                              ventanas_entrenamiento=int(len(f["entrenamiento"])),
                              ventanas_evaluacion=int(len(f["evaluacion"])),
                              estaciones_evaluacion=int(len(set(p.estacion[f["evaluacion"]])))))
    r = dict(esquema={c: {k: (list(v) if isinstance(v, tuple) else v) for k, v in e.items()}
                      for c, e in ESQUEMA.items()}, pasos=pasos)
    json.dump(r, open("resultados/e2/contrato_datos.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for s in pasos:
        print(s)
