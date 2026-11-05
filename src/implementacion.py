# %% [markdown]
# # Implementación de los tres modelos
#
# El código de los modelos está en `src/modelos.py` y la preparación de datos en `src/datos.py`. Todos los modelos
# comparten la misma entrada ($168 \times 4$), la misma pérdida (MSE), el mismo optimizador (Adam), lotes de 1024
# ventanas y parada temprana por el RMSE de validación (2024). La prueba (2025) no se usa en este capítulo.

# %%
import json, sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
fuente = open("src/modelos.py", encoding="utf-8").read()
print(fuente[fuente.index("class MLP"):fuente.index("MODELOS =")])

# %% [markdown]
# ## Control de la aleatoriedad
#
# Cada corrida fija la semilla de Python, NumPy y PyTorch (CPU y GPU) y activa el modo determinista de cuDNN
# (`modelos.fijar_semilla`). El orden de los lotes también sale de un generador con la misma semilla. Así, una
# corrida con la misma semilla y el mismo hardware reproduce el mismo resultado. Se comprobó repitiendo las 18
# corridas definitivas: los RMSE de validación coinciden en los cuatro decimales.
#
# ## Entorno de ejecución
#
# | Elemento | Valor |
# |---|---|
# | GPU | NVIDIA GeForce RTX 4060 Ti, 8 GB (CUDA 12.6) |
# | CPU y memoria | AMD Ryzen 7 8700F (8 núcleos), 32 GB de RAM |
# | Sistema | Windows 11 |
# | Entrenamiento | Python 3.11, PyTorch 2.14.1 (entorno `dl_env`) |
# | Análisis | pandas, statsmodels, scikit-learn y SciPy (entorno `ml_env`) |
#
# Las versiones exactas de cada librería están en `requirements.txt`.
#
# **Hiperparámetros fijos para los tres modelos:** optimizador Adam sin planificador de tasa de aprendizaje ni
# decaimiento de pesos, pérdida MSE, lotes de 1024, máximo 50 épocas y parada temprana con paciencia 5 sobre el RMSE
# de validación (se restauran los pesos de la mejor época).

# %% [markdown]
# ## Piloto: variabilidad entre semillas y número de corridas
#
# Se entrenó cada modelo con 5 semillas y la configuración inicial, evaluando solo en validación.

# %%
pil = pd.DataFrame(json.load(open("resultados/piloto.json")))
res_pil = pil.groupby("modelo").agg(RMSE_val_media=("rmse_val", "mean"), RMSE_val_de=("rmse_val", "std"),
                                    epocas=("epocas", "mean"), segundos=("segundos", "mean"))
res_pil["CV %"] = 100 * res_pil.RMSE_val_de / res_pil.RMSE_val_media
print(res_pil.round(4))
ancho = pil.pivot(index="semilla", columns="modelo", values="rmse_val")
sig_d = {f"{a} − {b}": (ancho[a] - ancho[b]).std(ddof=1)
         for i, a in enumerate(ancho) for b in list(ancho)[i + 1:]}
print("\nDesviación estándar de la diferencia pareada por semilla (σ_d):")
for i, a in enumerate(ancho):
    for b in list(ancho)[i + 1:]:
        dif = ancho[a] - ancho[b]
        print(f"  {a} − {b}: σ_d = {dif.std(ddof=1):.4f} | δ/σ_d = {0.02 / dif.std(ddof=1):.1f} | "
              f"Shapiro-Wilk p = {stats.shapiro(dif).pvalue:.2f}")

# %% [markdown]
# Para detectar una diferencia de $\delta = 0.02$ m/s con $\alpha = 0.05$ (bilateral) y potencia del 80 %, el número
# de semillas por modelo es
#
# $$R \approx \frac{(z_{0.975} + z_{0.80})^2\, \sigma_d^2}{\delta^2} = \frac{(1.96 + 0.84)^2\, \sigma_d^2}{\delta^2}.$$
#
# La prueba de Shapiro-Wilk no rechaza la normalidad de las diferencias pareadas, pero con 5 pares tiene muy poca
# potencia; por eso la inferencia final se apoya también en pruebas no paramétricas y en bootstrap.

# %%
delta = 0.02
sd_max = max(sig_d.values())
R = int(np.ceil((1.96 + 0.84) ** 2 * sd_max ** 2 / delta ** 2))
R_final = max(R, 6)
print(f"σ_d máxima = {sd_max:.4f} → R = {R} → se usan R = {R_final} semillas (mínimo 6 exigido)")
print(f"Diferencia mínima detectable con R = {R_final}: {(1.96 + 0.84) * sd_max / np.sqrt(R_final):.4f} m/s")

# %% [markdown]
# El resultado se registró en el [prerregistro](https://github.com/BetzyW/Deep-Learning-Wind/blob/main/prerregistro.md)
# antes del ajuste de hiperparámetros y de evaluar el conjunto de prueba: modelos, métrica, margen, número de semillas y pruebas estadísticas quedaron fijados de antemano.
#
# ## Ajuste de hiperparámetros con el mismo presupuesto
#
# Búsqueda aleatoria de 20 configuraciones por modelo (`src/ajuste.py`), cada una con la misma semilla, máximo 30
# épocas y paciencia 3. Darle el mismo número de intentos a cada arquitectura evita favorecer a una por haberla
# ajustado más.

# %%
aj = pd.DataFrame(json.load(open("resultados/ajuste.json")))
print(aj.groupby("modelo").agg(configuraciones=("rmse_val", "size"), mejor=("rmse_val", "min"),
                               mediana=("rmse_val", "median"), peor=("rmse_val", "max"),
                               minutos=("segundos", lambda s: s.sum() / 60)).round(4))
print("\nHiperparámetros elegidos:"); print(json.dumps(json.load(open("configs/hiperparametros.json")), indent=1))
fig, ax = plt.subplots(figsize=(7, 3))
for m, g in aj.groupby("modelo"):
    ax.scatter(g.parametros, g.rmse_val, label=m, alpha=.7)
ax.set(xscale="log", xlabel="parámetros", ylabel="RMSE validación (m/s)", title="Configuraciones probadas")
ax.legend(); plt.show()

# %% [markdown]
# ## Corridas definitivas: curvas de entrenamiento y validación
#
# Líneas continuas: RMSE de validación por época. Líneas punteadas: RMSE de entrenamiento (medido durante la época,
# con el dropout activo).

# %%
co = pd.DataFrame(json.load(open("resultados/corridas.json")))
fig, ax = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
for a_, (m, g) in zip(ax, co.groupby("modelo")):
    for i, r in enumerate(g.itertuples()):
        col = f"C{i}"
        a_.plot(range(1, len(r.historia) + 1), r.historia, color=col, alpha=.8)
        a_.plot(range(1, len(r.historia_train) + 1), r.historia_train, color=col, ls=":", alpha=.8)
    a_.set(title=m, xlabel="época"); ax[0].set_ylabel("RMSE (m/s)")
plt.tight_layout(); plt.show()
co["rmse_train_final"] = [h[int(np.argmin(v))] for h, v in zip(co.historia_train, co.historia)]
res = co.groupby("modelo").agg(RMSE_train=("rmse_train_final", "mean"), RMSE_val=("rmse_val", "mean"),
                               de_val=("rmse_val", "std"), epocas=("epocas", "mean"), segundos=("segundos", "mean"),
                               parametros=("parametros", "first"))
res["brecha val − train"] = res.RMSE_val - res.RMSE_train
print(res.round(4))

# %% [markdown]
# El error de validación es **mayor** que el de entrenamiento en los tres modelos, pero la brecha no crece con las
# épocas: la curva de validación se aplana en pocas épocas y la parada temprana corta antes de que empiece a subir.
# Parte de la brecha se explica por el desplazamiento entre años que mostró el EDA (validación cargada hacia
# noviembre y diciembre), no solo por sobreajuste. Ningún modelo muestra subajuste evidente: todos bajan mucho del
# error de la persistencia (0.963 m/s en validación).

# %% [markdown]
# ## Sensibilidad: con presión como quinta variable
#
# En el EDA se descartó la presión por su faltante no aleatorio. Para medir cuánto se pierde, se repitió el
# entrenamiento con la configuración `configs/datos_con_presion.json` (3 semillas por modelo, solo validación).
# Como esa configuración tiene menos estaciones, la comparación es orientativa.

# %%
import os
if os.path.exists("resultados/sensibilidad_presion.json"):
    se = pd.DataFrame(json.load(open("resultados/sensibilidad_presion.json")))
    comp = pd.DataFrame({"sin presión (4 variables)": co.groupby("modelo").rmse_val.mean(),
                         "con presión (5 variables)": se.groupby("modelo").rmse_val.mean()})
    print(comp.round(4))
