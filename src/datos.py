"""Ventanas deslizantes del panel de viento para los modelos de Deep Learning.

Cada muestra es una ventana de L horas con las variables de `columnas` (entrada) y la
velocidad observada H horas después del final de la ventana (objetivo).
La partición es por año: una ventana pertenece a un conjunto solo si toda su extensión,
desde la primera hora de entrada hasta la hora objetivo, cae dentro de ese conjunto.
Eso equivale a un embargo de L + H horas en cada frontera, sin fuga entre conjuntos.
"""
import json
import numpy as np
import pandas as pd


def cargar(config="configs/datos.json", panel=None):
    cfg = json.load(open(config, encoding="utf-8"))
    cols, L, H = cfg["columnas"], cfg["L"], cfg["H"]
    df = pd.read_csv(panel or cfg["panel"], usecols=["CodigoEstacion", "hora"] + cols,
                     dtype={"CodigoEstacion": str}, parse_dates=["hora"])
    df = df[~df.CodigoEstacion.isin(cfg.get("excluir_estaciones", []))]   # sensores dañados (ver EDA)

    def conjunto(anio):
        return np.where(anio <= cfg["entrenamiento_hasta"], 0,
                        np.where(anio == cfg["validacion"], 1, np.where(anio == cfg["prueba"], 2, -1)))

    series, tiempos, inicios, grupo, estacion, ofs = [], [], [], [], [], 0
    for cod, g in df.groupby("CodigoEstacion"):
        g = g.set_index("hora").sort_index()
        g = g[~g.index.duplicated()].asfreq("h")                       # rejilla horaria regular
        x = g[cols].ffill(limit=cfg["ffill_max_horas"]).to_numpy(np.float32)  # solo mira al pasado
        ok = ~np.isnan(x).any(1)
        n = len(x) - L - H + 1
        if n <= 0:
            continue
        c = np.concatenate([[0], np.cumsum(ok)])
        i = np.arange(n)
        valida = (c[i + L] - c[i] == L) & ok[i + L - 1 + H]           # entrada completa y objetivo observado
        anio = g.index.year.to_numpy()
        s_ini, s_obj = conjunto(anio[i]), conjunto(anio[i + L - 1 + H])
        valida &= (s_ini == s_obj) & (s_obj >= 0)                       # embargo: toda la ventana en un conjunto
        inicios.append(ofs + i[valida]); grupo.append(s_obj[valida])
        estacion.append(np.full(valida.sum(), cod))
        series.append(np.nan_to_num(x)); tiempos.append(g.index.to_numpy()); ofs += len(x)

    X = np.concatenate(series)
    inicios, grupo, estacion = np.concatenate(inicios), np.concatenate(grupo), np.concatenate(estacion)
    # escalado con las filas que alimentan ventanas de entrenamiento únicamente
    filas_tr = np.unique((inicios[grupo == 0][:, None] + np.arange(L)).ravel())
    mu, sd = X[filas_tr].mean(0), X[filas_tr].std(0) + 1e-6
    return dict(X=(X - mu) / sd, y=X[:, 0].copy(), tiempo=np.concatenate(tiempos), mu=mu, sd=sd, L=L, H=H, cols=cols,
                idx={n: inicios[grupo == k] for k, n in enumerate(["entrenamiento", "validacion", "prueba"])},
                estacion={n: estacion[grupo == k] for k, n in enumerate(["entrenamiento", "validacion", "prueba"])})


class Lotes:
    """Arma lotes (B, L, d) directamente en la GPU a partir de índices de inicio de ventana."""

    def __init__(self, d, dispositivo="cuda"):
        import torch  # solo se necesita para entrenar; el EDA corre sin PyTorch
        self.torch = torch
        self.X = torch.tensor(d["X"], device=dispositivo)
        self.y = torch.tensor(d["y"], device=dispositivo)
        self.off = torch.arange(d["L"], device=dispositivo)
        self.L, self.H, self.dev = d["L"], d["H"], dispositivo

    def __call__(self, idx):
        idx = self.torch.as_tensor(idx, device=self.dev)
        return self.X[idx[:, None] + self.off], self.y[idx + self.L - 1 + self.H]


if __name__ == "__main__":
    import sys, time
    t = time.time()
    d = cargar(panel=sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"Preparación: {time.time() - t:.0f} s")
    for n, ix in d["idx"].items():
        print(f"{n:13s} ventanas {len(ix):>9,} | estaciones {len(set(d['estacion'][n])):3d}")
    # verificación: persistencia (v_t como pronóstico de v_{t+H}) en unidades originales
    for n in ["validacion", "prueba"]:
        ix = d["idx"][n]
        pred, obs = d["y"][ix + d["L"] - 1], d["y"][ix + d["L"] - 1 + d["H"]]
        print(f"Persistencia {n}: RMSE {np.sqrt(((pred - obs) ** 2).mean()):.3f} m/s")
    x, y = Lotes(d)(d["idx"]["entrenamiento"][:1024])
    print("Lote de prueba:", tuple(x.shape), tuple(y.shape), x.device)
