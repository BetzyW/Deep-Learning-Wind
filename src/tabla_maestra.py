"""Tabla maestra de experimentos del E3 (entregable 8 del enunciado): una fila por corrida, con el modelo original, los
13 modelos del benchmark (Top 10 y modelos base del E1), los baselines, las variantes de la ablación y los tres
conjuntos de datos adicionales, con sus métricas por ejecución, tiempos, costo computacional y semillas.

Columnas: conjunto, modelo, grupo (original, ablación, top10, base_e1, baseline), origen, semilla (vacía en los
modelos deterministas), n (ventanas), rmse, mae, rmse_parada, rmse_train, epocas, segundos (entrenamiento),
parametros, y para el panel del IDEAM mflops_ventana y us_ventana_cpu (latencia en CPU con lote 1 024, medida igual
para todos en `costo_inferencia.py --e3`). Unidades del RMSE: m/s (IDEAM, NOAA), kW (SDWPF), °C (ETTh1).
Salida: resultados/e3/tabla_maestra.csv
Uso: python src/tabla_maestra.py      (ml_env)
"""
import glob, os
import pandas as pd

TOP10 = ["Transformer", "KAN", "TCN", "STGCN", "Chronos", "GRU", "Informer", "DLinear", "Moirai", "PatchTST"]
BASE_E1 = ["MLP", "CNN", "LSTM"]
COLS = ["conjunto", "modelo", "grupo", "origen", "semilla", "n", "rmse", "mae", "rmse_parada", "rmse_train", "epocas",
        "segundos", "parametros"]


def grupo(m):
    return ("original" if m == "PADRE" else "ablacion" if m in ("V1", "V3", "V4", "V5") else
            "top10" if m in TOP10 else "base_e1" if m in BASE_E1 else "baseline")


if __name__ == "__main__":
    partes = []
    # panel del IDEAM: benchmarks del E2 (10 semillas), PADRE-TCN y su ablación
    for m in TOP10 + BASE_E1:
        d = pd.read_csv(f"resultados/e2/finales/{m}.csv")
        partes.append(d[d.semilla.between(1000, 1009)].assign(conjunto="ideam", modelo=m))
    for v in ["V6", "V1", "V3", "V4", "V5"]:
        d = pd.read_csv(f"resultados/e3/confirmatorias/{v}.csv")
        partes.append(d.assign(conjunto="ideam", modelo="PADRE" if v == "V6" else v))
    b = pd.read_csv("resultados/e2/baselines_prueba_por_origen.csv")
    partes.append(b.assign(conjunto="ideam", semilla=pd.NA))
    # conjuntos adicionales
    for c in ["etth1", "sdwpf", "noaa_kansas"]:
        for ruta in sorted(glob.glob(f"resultados/e3/adicionales/{c}/*.csv")):
            nombre = os.path.basename(ruta)[:-4]
            if nombre in ("resumen",):
                continue
            d = pd.read_csv(ruta)
            if nombre == "baselines":
                d = d.assign(semilla=pd.NA)
            partes.append(d.assign(conjunto=c))
    t = pd.concat(partes, ignore_index=True)
    t["modelo"] = t.modelo.str.replace("_", " ")
    t["grupo"] = t.modelo.map(grupo)
    t = t.reindex(columns=COLS)
    costo = pd.read_csv("resultados/e3/costo_cpu.csv").set_index("modelo")
    ideam = t.conjunto == "ideam"
    t.loc[ideam, "mflops_ventana"] = t.loc[ideam, "modelo"].map(costo.mflops_ventana)
    t.loc[ideam, "us_ventana_cpu"] = t.loc[ideam, "modelo"].map(costo.ms_ventana_lote1024 * 1e3)
    t = t.sort_values(["conjunto", "grupo", "modelo", "origen", "semilla"]).reset_index(drop=True)
    t.to_csv("resultados/e3/tabla_maestra.csv", index=False)
    print(t.groupby(["conjunto", "grupo"]).size().to_string())
    print(len(t), "filas")
