"""Tabla de puntajes para elegir el Top 10 (sección 3.2 del enunciado). Uso: python puntaje.py → puntajes_top10.csv

Cinco criterios, cada uno llevado a [0, 1], con pesos iguales (0.2) fijados antes de calcularlos:
  1. Citas normalizadas por antigüedad: log(1 + citas / años desde la publicación), escalado min-max.
  2. Recencia: (año − 2014) / (2024 − 2014), recortado a [0, 1].
  3. Desempeño reportado en datos comparables: número de estudios elegibles de viento (tamizaje_e2.py) cuyo TÍTULO
     nombra la arquitectura, es decir, donde es el modelo propuesto o protagonista; log-escalado. Es un indicador
     de que la arquitectura rinde en viento según sus autores, no una medida directa del error.
  4. Implementación disponible: 1 = código oficial y biblioteca mantenida; 0.5 = solo código oficial o de terceros.
  5. Representatividad de la familia: 1 para el mejor de su familia según los criterios 1–4, 0.5 el segundo,
     0.25 el tercero, etc. Favorece la diversidad que exige el enunciado (al menos cuatro familias).
Se excluyen los tres modelos base del E1 (MLP, LSTM y CNN 1D), que entran al benchmark aparte.
"""
import json, math
import pandas as pd

ANIO_ACTUAL = 2026
# arquitectura: (familia, clave en frecuencia_arquitecturas.csv, implementación, repositorio o biblioteca)
META = {
    "Transformer": ("Atención", "Transformer", 1, "PyTorch nn.Transformer"),
    "Informer": ("Atención", "Informer", 1, "zhouhaoyi/Informer2020; neuralforecast"),
    "Autoformer": ("Atención", "Autoformer", 1, "thuml/Autoformer; neuralforecast"),
    "FEDformer": ("Atención", "FEDformer", 1, "MAZiqing/FEDformer; neuralforecast"),
    "PatchTST": ("Atención", "PatchTST", 1, "yuqinie98/PatchTST; neuralforecast"),
    "iTransformer": ("Atención", "iTransformer", 1, "thuml/iTransformer; neuralforecast"),
    "TFT": ("Atención", "TFT", 1, "google-research; pytorch-forecasting; neuralforecast"),
    "TCN": ("Convolucional", "TCN", 1, "locuslab/TCN; neuralforecast"),
    "WaveNet": ("Convolucional", "WaveNet", 0.5, "solo implementaciones de terceros"),
    "TimesNet": ("Convolucional", "TimesNet", 1, "thuml/Time-Series-Library; neuralforecast"),
    "N-BEATS": ("MLP profundo", "N-BEATS", 1, "ServiceNow/N-BEATS; neuralforecast"),
    "N-HiTS": ("MLP profundo", "N-HiTS", 1, "neuralforecast (oficial)"),
    "DLinear": ("MLP profundo", "DLinear", 1, "cure-lab/LTSF-Linear; neuralforecast"),
    "TSMixer": ("MLP profundo", "TSMixer", 1, "google-research; neuralforecast"),
    "TiDE": ("MLP profundo", None, 1, "google-research; neuralforecast"),
    "KAN": ("MLP profundo", "KAN", 1, "KindXiaoming/pykan; neuralforecast"),
    "GRU": ("Recurrente", "GRU", 1, "PyTorch nn.GRU"),
    "xLSTM": ("Recurrente", "xLSTM", 0.5, "NX-AI/xlstm"),
    "DeepAR": ("Recurrente", "DeepAR", 1, "GluonTS; pytorch-forecasting"),
    "Mamba": ("Espacio de estados", "Mamba/SSM", 0.5, "state-spaces/mamba (núcleos CUDA para Linux)"),
    "STGCN": ("Grafos", "GNN/GCN", 1, "VeritasYin/STGCN_IJCAI-18; torch-geometric-temporal"),
    "Graph WaveNet": ("Grafos", "GNN/GCN", 0.5, "nnzhan/Graph-WaveNet"),
    "Chronos": ("Preentrenado", "Chronos", 1, "amazon-science/chronos-forecasting"),
    "TimesFM": ("Preentrenado", "TimesFM", 1, "google-research/timesfm"),
    "Moirai": ("Preentrenado", "Moirai", 1, "SalesforceAIResearch/uni2ts"),
    "Lag-Llama": ("Preentrenado", "Lag-Llama", 0.5, "time-series-foundation-models/lag-llama"),
    "Time-LLM": ("Preentrenado", "LLM (adaptado)", 0.5, "KimMeen/Time-LLM"),
    "TimeGrad": ("Generativo probabilístico", "Difusión", 0.5, "zalandoresearch/pytorch-ts"),
}


def minmax(s):
    return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0


if __name__ == "__main__":
    orig = json.load(open("originales.json", encoding="utf-8"))
    frec = pd.read_csv("frecuencia_arquitecturas.csv").set_index("arquitectura")
    filas = []
    for k, (fam, clave, impl, repo) in META.items():
        o = orig[k]
        edad = max(1, ANIO_ACTUAL - o["anio"] + 1)
        filas.append(dict(modelo=k, familia=fam, anio=o["anio"], venue=o["venue"], doi=o["doi"] or f"arXiv:{o['arxiv']}",
                          citas=o["citas"], citas_por_anio=o["citas"] / edad,
                          estudios_viento=int(frec.loc[clave, "estudios"]) if clave in frec.index else 0,
                          protagonista_viento=int(frec.loc[clave, "en_titulo"]) if clave in frec.index else 0,
                          implementacion=impl, repositorio=repo))
    t = pd.DataFrame(filas)
    t["c1_citas"] = minmax(t.citas_por_anio.map(math.log1p))
    t["c2_recencia"] = ((t.anio - 2014) / 10).clip(0, 1)
    t["c3_desempeno_viento"] = minmax(t.protagonista_viento.map(math.log1p))
    t["c4_implementacion"] = t.implementacion
    base = 0.25 * (t.c1_citas + t.c2_recencia + t.c3_desempeno_viento + t.c4_implementacion)
    t["c5_representatividad"] = 0.5 ** base.groupby(t.familia).rank(ascending=False, method="first").sub(1)
    t["puntaje"] = 0.2 * (t.c1_citas + t.c2_recencia + t.c3_desempeno_viento + t.c4_implementacion
                          + t.c5_representatividad)
    t = t.sort_values("puntaje", ascending=False).reset_index(drop=True)
    t.index += 1
    t["top10"] = t.index <= 10
    t.round(3).to_csv("puntajes_top10.csv", index_label="rango")
    cols = ["modelo", "familia", "anio", "citas_por_anio", "protagonista_viento", "c1_citas", "c2_recencia",
            "c3_desempeno_viento", "c4_implementacion", "c5_representatividad", "puntaje"]
    print(t[cols].round(3).to_string())
    top = t[t.top10]
    print(f"\nFamilias en el Top 10: {top.familia.nunique()} → {sorted(top.familia.unique())}")
