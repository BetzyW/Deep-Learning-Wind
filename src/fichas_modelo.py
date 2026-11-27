"""Fichas de modelo según la plantilla de la sección 11.3 del enunciado, armadas desde los archivos de resultados.

Cada ficha junta: identificación, preentrenamiento, arquitectura y costo (`costo_inferencia.py`), hiperparámetros
finales con el espacio explorado, arquitectura seleccionada (si hubo búsqueda), entrenamiento, cómputo, resultados por
ejecución e incidencias. Lo único escrito a mano son la familia, la referencia, la fuente de pesos y las incidencias,
que vienen de los capítulos de modelos y del preregistro.
Salida: book/tablas/e2_fichas.md
Uso: python src/fichas_modelo.py      (ml_env)
"""
import json, os, subprocess
import numpy as np
import pandas as pd

E, B = "resultados/e2", "resultados/e2/busqueda"
INFO = {
    "Transformer": ("Atención", "vaswani2017", "ninguno",
                    "Corridas finales repartidas entre una A100 y una L4 de Colab. La elegida es la referencia "
                    "(64, 64, 64) / (0, 1, 0)."),
    "KAN": ("MLP profundo", "liu2024kan", "ninguno", "Corridas finales repartidas entre una A100 y una L4 de Colab."),
    "TCN": ("Convolucional", "bai2018", "ninguno", "Toda en el PC (RTX 4060 Ti). Modelo de la ablación (enmienda 7)."),
    "STGCN": ("Grafos", "yu2018", "ninguno", "Subgrafo de la estación y sus 8 vecinas más cercanas, porque el panel no "
              "es una red sincronizada (enmienda 6)."),
    "Chronos": ("Preentrenado", "ansari2024", "`amazon/chronos-t5-small` (Hugging Face, Apache 2.0), congelado",
                "La representación congelada se calculó en Colab (`embeber.py`); solo se entrena el cabezal."),
    "GRU": ("Recurrente", "cho2014", "ninguno", "En el PC tardaba de 20 a 77 min por corrida; las fases y la validación "
            "con origen móvil se hicieron en Colab."),
    "Informer": ("Atención", "zhou2021", "ninguno", "Algunas configuraciones del ajuste previo fallaron por memoria y se "
                 "descartaron sin resultados; los lotes se procesan en trozos de 256 (enmienda 5)."),
    "DLinear": ("MLP profundo (paper guía)", "zeng2023", "ninguno", "Paper guía; replicación parcial en el capítulo de "
                "resultados."),
    "Moirai": ("Preentrenado", "woo2024", "`Salesforce/moirai-1.0-R-small` (Hugging Face, CC BY-NC 4.0), congelado",
               "Parche de 32 h, porque `\"auto\"` falla con 168 h de contexto. Representación calculada en Colab."),
    "PatchTST": ("Atención", "nie2023", "ninguno", "Sustituye a Mamba (enmienda 4). El primer ajuste previo tenía un "
                 "error de escala y se descartó entero antes de elegir hiperparámetros "
                 "(`PatchTST_hp_descartado_error_escala.csv`)."),
    "MLP": ("Modelo base del E1", "rumelhart1986", "ninguno", "—"),
    "CNN": ("Modelo base del E1", "lecun1998", "ninguno", "—"),
    "LSTM": ("Modelo base del E1", "hochreiter1997", "ninguno", "—"),
}
ESPACIO_HP = "lr log-uniforme en [1e-4, 3e-3]; wd ∈ {0, 1e-5, 1e-4}; lote ∈ {512, 1 024, 2 048}"


def ficha(m, costo, commit):
    fam, ref, pesos, inc = INFO[m]
    c = json.load(open(f"{E}/finales/{m}_configuracion.json"))
    hp, arq = c["hp"], c["arquitectura"]
    fin = pd.read_csv(f"{E}/finales/{m}.csv")
    decl = fin[fin.declarada_e2 == 1]
    hpcsv = pd.read_csv(f"{B}/{m}_hp.csv")
    k = costo.loc[m]
    hw = [json.loads(l) for l in open(f"{B}/hardware.jsonl", encoding="utf-8")]
    gpus = sorted({h["gpu"] for h in hw if h["modelo"] == m and h["etapa"] == "corridas finales"})
    malas = int((~np.isfinite(fin.rmse)).sum())
    filas = [
        ("Identificación", f"{m}, familia {fam}; {{cite}}`{ref}`; implementación propia en `src/modelos_e2.py` "
                           f"(commit `{commit}`)"),
        ("Preentrenamiento", pesos),
        ("Arquitectura", f"`{int(k.parametros):,}`".replace(",", " ") + f" parámetros entrenables; "
                         f"`{k.mflops_ventana:.2f}` MFLOPs por ventana de 168 h"),
        ("Hiperparámetros finales", f"lr = `{hp['lr']:.2e}`, wd = `{hp['wd']:g}`, lote = `{hp['bs']}`; espacio: "
                                    f"{ESPACIO_HP}; T = `{len(hpcsv)}` intentos con el 10 % de las ventanas"),
        ("Arquitectura seleccionada",
         (f"`{arq['id']}` (anchos y patrón de dropout); `{len(arq['empatadas'])}` finalistas indistinguibles a "
          f"menos de δ/2; sesgo del ganador y frente de Pareto en el capítulo de resultados") if arq else
         "Publicada por los autores; sin búsqueda (sección 14)"),
        ("Entrenamiento", f"Adam, sin scheduler, MSE, float32; máximo 50 épocas con paciencia 5 sobre las últimas "
                          f"2 160 h del entrenamiento; épocas efectivas: mediana `{decl.epocas.median():.0f}` "
                          f"(de `{decl.epocas.min()}` a `{decl.epocas.max()}`)"),
        ("Cómputo", f"Corridas finales en {', '.join(g.replace('NVIDIA ', '') for g in gpus)}; "
                    f"`{fin.segundos.sum() / 3600:.1f}` h para las {len(fin)} corridas (10 semillas × 4 orígenes); "
                    f"ajuste previo `{hpcsv.segundos.sum() / 3600:.2f}` h; latencia en CPU `{k.ms_ventana_lote1:.2f}` ms "
                    f"por ventana (lote 1) y `{k.ms_ventana_lote1024:.4f}` ms (lote 1 024); memoria pico de GPU no "
                    f"medida"),
        ("Resultados por ejecución", f"`resultados/e2/finales/{m}.csv` (RMSE y MAE de prueba, RMSE de parada y de "
                                     f"entrenamiento por semilla y origen); RMSE de prueba por semilla "
                                     f"`{decl.rmse.min():.3f}`–`{decl.rmse.max():.3f}`"),
        ("Incidencias", (f"{malas} corridas con NaN. " if malas else "Ninguna corrida con NaN ni excluida. ") + inc),
    ]
    return f"### {m}\n\n| Campo | Contenido |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in filas)


if __name__ == "__main__":
    costo = pd.read_csv(f"{E}/costo_cpu.csv").set_index("modelo")
    commit = subprocess.run(["git", "log", "-1", "--format=%h", "--", "src/modelos_e2.py"], capture_output=True,
                            text=True).stdout.strip()
    texto = "\n\n".join(ficha(m, costo, commit) for m in INFO)
    open("book/tablas/e2_fichas.md", "w", encoding="utf-8").write(texto + "\n")
    print(texto[:3000])
