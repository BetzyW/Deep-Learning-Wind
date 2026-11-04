# %% [markdown]
# # Benchmark, baseline y pruebas estadísticas
#
# Las predicciones de 2025 se cargan de `resultados/predicciones_prueba.npz` (generadas por `src/corridas.py`). Aquí
# no se entrena nada: solo se evalúa el conjunto de prueba, que estuvo congelado hasta este punto.

# %%
import json, sys, warnings
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import bds
sys.path.insert(0, "src")
from datos import cargar

warnings.simplefilter("ignore")
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
d = cargar()
L, H = d["L"], d["H"]
ix = d["idx"]["prueba"]
y = d["y"][ix + L - 1 + H]                      # velocidad observada en t + 24
pers = d["y"][ix + L - 1]                       # persistencia: v_t
est = d["estacion"]["prueba"]
hora = d["tiempo"][ix + L - 1 + H]
P = np.load("resultados/predicciones_prueba.npz")
corr = pd.DataFrame(json.load(open("resultados/corridas.json")))
MOD = ["MLP", "LSTM", "CNN"]
semillas = sorted({int(k.split("_")[1]) for k in P.files})
print(f"{len(y):,} ventanas de prueba | {len(set(est))} estaciones | {len(semillas)} semillas por modelo")

# %% [markdown]
# ## Tabla comparativa
#
# Cada modelo se entrenó con $R$ semillas. Se reporta la media y la desviación estándar entre semillas de cada
# métrica, y el MASE como el cociente entre el MAE del modelo y el de la persistencia a 24 h.

# %%
def metricas(p):
    e = p - y
    return dict(RMSE=np.sqrt(np.mean(e ** 2)), MAE=np.mean(np.abs(e)), R2=1 - np.mean(e ** 2) / np.var(y))

mae_p = np.mean(np.abs(pers - y))
filas = []
for m in MOD:
    met = pd.DataFrame([metricas(P[f"{m}_{s}"]) for s in semillas])
    met["MASE"] = met.MAE / mae_p
    c = corr[corr.modelo == m]
    filas.append({"Modelo": m, **{f"{k} media": met[k].mean() for k in met},
                  "RMSE de": met.RMSE.std(ddof=1), "Parámetros": int(c.parametros.iloc[0]),
                  "Épocas": c.epocas.mean(), "Segundos de entrenamiento": c.segundos.mean()})
filas.append({"Modelo": "Persistencia", **{f"{k} media": v for k, v in metricas(pers).items()}, "MASE media": 1.0,
              "RMSE de": 0.0, "Parámetros": 0, "Épocas": 0, "Segundos de entrenamiento": 0})
tabla = pd.DataFrame(filas).set_index("Modelo")
tabla.round(4)

# %% [markdown]
# ## Pronóstico promedio por modelo
#
# Para las pruebas por estación y por hora se usa el **promedio de las $R$ semillas** de cada modelo, que es el
# pronóstico que se pondría en producción; la variabilidad entre semillas ya quedó en la tabla anterior.

# %%
pm = {m: np.mean([P[f"{m}_{s}"] for s in semillas], axis=0) for m in MOD}
pm["Persistencia"] = pers
err2 = pd.DataFrame({m: (p - y) ** 2 for m, p in pm.items()})
err2["estacion"], err2["hora"] = est, hora
rmse_est = err2.groupby("estacion")[list(pm)].mean() ** 0.5
print(f"RMSE por estación ({len(rmse_est)} estaciones), mediana:")
print(rmse_est.median().round(4).to_string())
print("\nRMSE del promedio de semillas:", {m: round(float(np.sqrt(err2[m].mean())), 4) for m in pm})

# %% [markdown]
# ## Ranking y prueba de Friedman (estaciones como bloques)
#
# La estación es la unidad de independencia declarada a priori, así que cada estación es un bloque y se comparan los
# rangos del RMSE de los cuatro pronósticos.

# %%
fr = stats.friedmanchisquare(*[rmse_est[m] for m in pm])
rangos = rmse_est.rank(axis=1).mean().sort_values()
print(f"Friedman: chi2 = {fr.statistic:.2f}, p = {fr.pvalue:.2e}")
print("Rango medio (1 = mejor):"); print(rangos.round(2).to_string())

# %% [markdown]
# ### Comparaciones por pares: Wilcoxon con corrección de Holm, diferencia de RMSE y tamaño del efecto
#
# Para cada par se usan las diferencias de RMSE por estación. El tamaño del efecto es la $d$ de Cohen para datos
# pareados (media de las diferencias sobre su desviación estándar) y el intervalo de confianza al 95 % de la
# diferencia media sale de un bootstrap por estaciones.

# %%
pares = [(a, b) for i, a in enumerate(pm) for b in list(pm)[i + 1:]]
rng = np.random.default_rng(0)
filas = []
for a, b in pares:
    dif = (rmse_est[a] - rmse_est[b]).to_numpy()
    boot = [rng.choice(dif, len(dif)).mean() for _ in range(5000)]
    filas.append({"Par": f"{a} − {b}", "ΔRMSE medio": dif.mean(), "IC 95 % inf": np.percentile(boot, 2.5),
                  "IC 95 % sup": np.percentile(boot, 97.5), "d de Cohen": dif.mean() / dif.std(ddof=1),
                  "p Wilcoxon": stats.wilcoxon(dif).pvalue})
wil = pd.DataFrame(filas).set_index("Par")
wil["p Holm"] = multipletests(wil["p Wilcoxon"], method="holm")[1]
wil["Significativo"] = wil["p Holm"] < 0.05
wil.round(4)

# %% [markdown]
# Supuesto de normalidad de las diferencias (relevante para la $d$ de Cohen y la prueba t de la equivalencia):

# %%
for a, b in pares:
    dif = rmse_est[a] - rmse_est[b]
    print(f"{a} − {b}: Shapiro-Wilk p = {stats.shapiro(dif).pvalue:.3f}")
fig, ax = plt.subplots(1, 3, figsize=(11, 3))
for a_, (a, b) in zip(ax, [(m, "Persistencia") for m in MOD]):
    stats.probplot(rmse_est[a] - rmse_est[b], plot=a_); a_.set_title(f"Q-Q {a} − persistencia", fontsize=9)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## Diebold–Mariano con corrección de Harvey–Leybourne–Newbold
#
# La prueba de Diebold–Mariano {cite}`diebold1995` compara la pérdida cuadrática de dos pronósticos hora a hora.
# Se promedia el diferencial de pérdida entre las estaciones activas en cada hora, para obtener una sola serie
# temporal, y se usa la varianza de largo plazo con $h - 1 = 23$ rezagos (el pronóstico es a 24 h, así que los errores
# consecutivos se solapan) y la corrección para muestras finitas de {cite}`harvey1997`.

# %%
def dm_hln(dl, h=24):
    dl = np.asarray(dl); T = len(dl); dbar = dl.mean(); u = dl - dbar
    gam = [u @ u / T] + [u[k:] @ u[:-k] / T for k in range(1, h)]
    v = (gam[0] + 2 * sum(gam[1:])) / T
    dm = dbar / np.sqrt(v)
    hln = dm * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    return hln, 2 * stats.t.sf(abs(hln), df=T - 1)

filas = []
for a, b in pares:
    serie = (err2[a] - err2[b]).groupby(err2.hora).mean().sort_index()
    est_, p = dm_hln(serie)
    filas.append({"Par": f"{a} − {b}", "Diferencial medio": serie.mean(), "DM-HLN": est_, "p": p})
dm = pd.DataFrame(filas).set_index("Par")
dm["p Holm"] = multipletests(dm.p, method="holm")[1]
dm.round(4)

# %% [markdown]
# Un diferencial negativo indica que el primer modelo del par tiene menor error cuadrático.

# %% [markdown]
# ## Equivalencia práctica (TOST) entre los dos mejores modelos
#
# Que una diferencia no sea significativa no prueba que los modelos sean iguales. La prueba TOST contrasta si la
# diferencia media de RMSE por estación cae dentro del margen declarado a priori, $\pm\delta = \pm 0.02$ m/s.

# %%
delta = 0.02
mejores = [m for m in rangos.index if m != "Persistencia"][:2]
dif = (rmse_est[mejores[0]] - rmse_est[mejores[1]]).to_numpy()
p_inf = stats.ttest_1samp(dif, -delta, alternative="greater").pvalue
p_sup = stats.ttest_1samp(dif, delta, alternative="less").pvalue
print(f"{mejores[0]} − {mejores[1]}: diferencia media {dif.mean():.4f} m/s | TOST p = {max(p_inf, p_sup):.4f}")
print("Equivalentes dentro de ±δ" if max(p_inf, p_sup) < 0.05 else "No se puede afirmar equivalencia dentro de ±δ")

# %% [markdown]
# ## Diagnóstico de los residuos del mejor modelo
#
# Si el pronóstico aprovechara toda la información, los residuos no tendrían estructura. Se aplican Ljung-Box
# (autocorrelación) y BDS (dependencia no lineal) a los residuos de cada estación con un tramo continuo suficiente.

# %%
mejor = mejores[0]
res_ = pd.DataFrame({"r": pm[mejor] - y, "estacion": est, "hora": hora})
filas = []
for c, g in res_.groupby("estacion"):
    s = g.set_index("hora").r.sort_index().asfreq("h").dropna()
    if len(s) < 1000:
        continue
    lb = acorr_ljungbox(s, lags=[24], return_df=True).lb_pvalue.iloc[0]
    b = bds(s.iloc[:3000].to_numpy(), max_dim=2)[1]
    filas.append((c, lb, float(np.atleast_1d(b)[0])))
diag = pd.DataFrame(filas, columns=["estacion", "p_LjungBox_24", "p_BDS_m2"])
print(f"{len(diag)} estaciones | Ljung-Box p < 0.05: {(diag.p_LjungBox_24 < .05).sum()} | "
      f"BDS p < 0.05: {(diag.p_BDS_m2 < .05).sum()}")
print(f"Sesgo medio de los residuos de {mejor}: {res_.r.mean():.3f} m/s")

# %% [markdown]
# ## Error por mes y por nivel de viento

# %%
mes = pd.DatetimeIndex(hora).month
por_mes = err2.groupby(mes)[list(pm)].mean() ** 0.5
nivel = pd.cut(y, [0, 2, 4, 6, 30], right=False, labels=["< 2", "2–4", "4–6", "≥ 6"])
por_nivel = err2.groupby(nivel, observed=True)[list(pm)].mean() ** 0.5
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
por_mes.plot(ax=ax[0], marker="o"); ax[0].set(xlabel="mes de 2025", ylabel="RMSE (m/s)", title="Por mes")
por_nivel.plot.bar(ax=ax[1], rot=0); ax[1].set(xlabel="velocidad observada (m/s)", title="Por nivel de viento")
plt.tight_layout(); plt.show()
print("Sesgo por nivel (predicho − observado):")
print(pd.DataFrame({m: (pm[m] - y) for m in pm}).groupby(nivel, observed=True).mean().round(3).to_string())
