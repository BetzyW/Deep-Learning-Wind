"""Bloque A del E3: ranking de los benchmarks antes y después de las correcciones, y congelamiento con hash.

1. Ranking antes y después. El E2 sacó sus conclusiones con las 6 semillas declaradas (1000–1005); el E3 exige 10
   (1000–1009). Se compara el RMSE medio por corrida (40 corridas = semillas × 4 orígenes de 2025) y el puesto de cada
   modelo con 6 y con 10 semillas. Los baselines deterministas no cambian.
2. Congelamiento: SHA-256 de cada archivo de resultados de los benchmarks y baselines (métricas por corrida,
   configuraciones elegidas y predicciones por ventana), más una huella global del conjunto. Desde aquí los
   benchmarks no se reajustan; solo se admiten correcciones de errores documentadas.
Salida: resultados/e3/ranking_antes_despues.csv y experiments/benchmarks_congelados.csv
Uso: python src/trazabilidad_e3.py      (ml_env)
"""
import glob, hashlib, os
import pandas as pd

FIN = "resultados/e2/finales"
MODELOS = ["Transformer", "KAN", "TCN", "STGCN", "Chronos", "GRU", "Informer", "DLinear", "Moirai", "PatchTST",
           "MLP", "CNN", "LSTM"]


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


if __name__ == "__main__":
    filas = []
    for m in MODELOS:
        d = pd.read_csv(f"{FIN}/{m}.csv")
        filas.append(dict(modelo=m, rmse_6=d[d.semilla.between(1000, 1005)].rmse.mean(),
                          rmse_10=d[d.semilla.between(1000, 1009)].rmse.mean()))
    b = pd.read_csv("resultados/e2/baselines_prueba_por_origen.csv")
    for m, g in b.groupby("modelo"):
        filas.append(dict(modelo=m, rmse_6=g.rmse.mean(), rmse_10=g.rmse.mean()))
    t = pd.DataFrame(filas)
    t["puesto_6"] = t.rmse_6.rank(method="min").astype(int)
    t["puesto_10"] = t.rmse_10.rank(method="min").astype(int)
    t["cambio_rmse"] = t.rmse_10 - t.rmse_6
    t = t.sort_values("puesto_10")
    t.to_csv("resultados/e3/ranking_antes_despues.csv", index=False)
    print(t.round(4).to_string(index=False))

    archivos = sorted(glob.glob(f"{FIN}/*.csv") + glob.glob(f"{FIN}/*_configuracion.json") +
                      glob.glob(f"{FIN}/*/*.npz") + glob.glob("resultados/e2/baselines_prueba*.csv"))
    h = pd.DataFrame([dict(archivo=a.replace(os.sep, "/"), bytes=os.path.getsize(a), sha256=sha256(a))
                      for a in archivos])
    global_ = hashlib.sha256("".join(h.archivo + h.sha256).encode()).hexdigest()
    h.to_csv("experiments/benchmarks_congelados.csv", index=False)
    print(f"{len(h)} archivos congelados; huella global {global_}")
    open("experiments/benchmarks_congelados.sha256", "w").write(global_ + "\n")
