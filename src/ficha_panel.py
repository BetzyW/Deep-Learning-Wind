"""Ficha del panel: hash SHA-256, tamaño, estaciones y rango de fechas (dato fijo para la Sección 1)."""
import hashlib, sys
import pandas as pd

PANEL = sys.argv[1] if len(sys.argv) > 1 else "datos/panel_multivariado_causal.csv"

h = hashlib.sha256()
with open(PANEL, "rb") as f:
    for bloque in iter(lambda: f.read(1 << 20), b""):
        h.update(bloque)
print("SHA-256:", h.hexdigest())

df = pd.read_csv(PANEL, dtype={"CodigoEstacion": str}, parse_dates=["hora"])
print(f"Filas: {len(df):,} | estaciones: {df.CodigoEstacion.nunique()} | departamentos: {df.depto.nunique()}")
print(f"Desde {df.hora.min()} hasta {df.hora.max()}")
print("Faltantes por columna (%):")
print((df[["vel", "dir_sin", "dir_cos", "temp", "pres"]].isna().mean() * 100).round(2).to_string())
print("Filas por año:")
print(df.hora.dt.year.value_counts().sort_index().to_string())
