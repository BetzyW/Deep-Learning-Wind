# %% [markdown]
# # Implementación de los tres modelos
#
# El código de los modelos está en `src/modelos.py` y la preparación de datos en `src/datos.py`. Todos los modelos
# comparten la misma entrada ($168 \times 4$), la misma pérdida (MSE), el mismo optimizador (Adam), lotes de 1024
# ventanas y parada temprana por el RMSE de validación (2024). La prueba (2025) no se usa en este capítulo.

# %%
import json, inspect, sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
sys.path.insert(0, "src")
import modelos
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
print(inspect.getsource(modelos.MLP)); print(inspect.getsource(modelos.LSTM)); print(inspect.getsource(modelos.CNN))

# %% [markdown]
# ## Control de la aleatoriedad
#
# Cada corrida fija la semilla de Python, NumPy y PyTorch (CPU y GPU) y activa el modo determinista de cuDNN
# (`modelos.fijar_semilla`). El orden de los lotes también sale de un generador con la misma semilla. Así, una
# corrida con la misma semilla y el mismo hardware reproduce el mismo resultado.

# %% [markdown]
# ## Piloto: variabilidad entre semillas y número de corridas
#
# Se entrenó cada modelo con 5 semillas y la configuración inicial, evaluando solo en validación.

# %%
pil = pd.DataFrame(json.load(open("resultados/piloto.json")))
print(pil.groupby("modelo").agg(RMSE_val_media=("rmse_val", "mean"), RMSE_val_de=("rmse_val", "std"),
                                epocas=("epocas", "mean"), segundos=("segundos", "mean")).round(4))
ancho = pil.pivot(index="semilla", columns="modelo", values="rmse_val")
sig_d = {f"{a} − {b}": (ancho[a] - ancho[b]).std(ddof=1)
         for i, a in enumerate(ancho) for b in list(ancho)[i + 1:]}
print("\nDesviación estándar de la diferencia pareada por semilla (σ_d):")
for k, v in sig_d.items():
    print(f"  {k}: {v:.4f}")

# %% [markdown]
# Para detectar una diferencia de $\delta = 0.02$ m/s con $\alpha = 0.05$ (bilateral) y potencia del 80 %, el número
# de semillas por modelo es
#
# $$R \approx \frac{(z_{0.975} + z_{0.80})^2\, \sigma_d^2}{\delta^2} = \frac{(1.96 + 0.84)^2\, \sigma_d^2}{\delta^2}.$$

# %%
delta = 0.02
sd_max = max(sig_d.values())
R = int(np.ceil((1.96 + 0.84) ** 2 * sd_max ** 2 / delta ** 2))
R_final = max(R, 6)
print(f"σ_d máxima = {sd_max:.4f} → R = {R} → se usan R = {R_final} semillas (mínimo 6 exigido)")
print(f"Diferencia mínima detectable con R = {R_final}: {(1.96 + 0.84) * sd_max / np.sqrt(R_final):.4f} m/s")

# %% [markdown]
# El resultado se registró en `prerregistro.md` antes de las corridas definitivas.
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
# ## Corridas definitivas: curvas de aprendizaje

# %%
co = pd.DataFrame(json.load(open("resultados/corridas.json")))
fig, ax = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
for a_, (m, g) in zip(ax, co.groupby("modelo")):
    for h in g.historia:
        a_.plot(range(1, len(h) + 1), h, alpha=.6)
    a_.set(title=m, xlabel="época"); ax[0].set_ylabel("RMSE validación")
plt.tight_layout(); plt.show()
print(co.groupby("modelo").agg(RMSE_val=("rmse_val", "mean"), de=("rmse_val", "std"), epocas=("epocas", "mean"),
                               segundos=("segundos", "mean"), parametros=("parametros", "first")).round(4))
