"""Techo de desempeño, potencia y componentes de varianza del E2 (secciones 12.1, 9.2 y 13.2 del enunciado).

* Techo (12.1): dos cotas sobre el RMSE de prueba de 2025.
  - Conjunto de modelos diversos: promedio de los pronósticos de los modelos de Deep Learning entrenados desde cero y de
    los dos clásicos fuertes (Ridge y gradient boosting). Es una cota alcanzable, sugerida por el enunciado.
  - Oráculo por estación: en cada estación, el mejor modelo elegido con la propia prueba. Es optimista (mira la
    prueba) y solo sirve como cota inferior del error.
  γ = (s_techo − s_mejor) / (s_techo − s_base), con s = RMSE, base = persistencia y mejor = el de menor RMSE agregado:
  la fracción del margen total que aún falta cubrir (γ ≤ 0 significa que el mejor modelo ya alcanzó esa cota).
  Además, la relación entre la predictibilidad intrínseca de cada estación (Ω de la entropía de permutación) y el
  RMSE del mejor modelo relativo a la persistencia.
* Potencia (9.2): con la prueba principal (Wilcoxon por estación, 62 estaciones, Holm con m = 10), diferencia mínima
  detectable al 80 % a partir de la desviación de las diferencias por estación entre el mejor modelo y Ridge; y el
  error estándar del RMSE medio de semillas con R = 6 a partir de la dispersión entre semillas de las corridas finales.
* Componentes de varianza (13.2): ANOVA del RMSE de cada corrida final (modelo × origen × semilla) con modelo, origen y
  su interacción; el residuo es la variación entre semillas.
Salida: resultados/e2/techo_potencia.json
Uso: python src/techo_potencia.py      (ml_env)
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from estadistica_e2 import ALFA, SEM_E2, cargar  # noqa: E402

DESDE_CERO = ["MLP", "CNN", "LSTM", "GRU", "TCN", "Transformer", "KAN", "Informer", "DLinear", "PatchTST", "STGCN"]
CLASICOS = ["Ridge", "Gradient_boosting"]
PREENTRENADOS = ["Chronos", "Moirai"]


def rmse(d):
    return float(np.sqrt(((d.pred - d.y) ** 2).mean()))


if __name__ == "__main__":
    from origen_movil import leer_config, preparar_panel
    p = preparar_panel(leer_config())
    modelos = DESDE_CERO + PREENTRENADOS + CLASICOS + ["Persistencia"]
    datos = {m: cargar(p, m).dropna(subset=["y"]).set_index("tau") for m in modelos}
    comun = datos["Persistencia"].index
    for m in modelos:
        comun = comun.intersection(datos[m].index)
    datos = {m: d.loc[comun] for m, d in datos.items()}
    res = {}

    # techo
    s = {m: rmse(datos[m]) for m in modelos}
    mejor = min(DESDE_CERO + PREENTRENADOS, key=s.get)
    conj = datos["Persistencia"][["estacion", "y"]].copy()
    conj["pred"] = np.mean([datos[m].pred.to_numpy() for m in DESDE_CERO + CLASICOS], 0)
    s_conj = rmse(conj)
    est = pd.DataFrame({m: datos[m].assign(e2=(datos[m].pred - datos[m].y) ** 2).groupby("estacion").e2.agg(["sum",
                        "size"])["sum"] for m in modelos})
    n_est = datos["Persistencia"].groupby("estacion").size()
    s_oraculo = float(np.sqrt(est[DESDE_CERO + PREENTRENADOS + CLASICOS].min(1).sum() / n_est.sum()))
    s_base = s["Persistencia"]
    gamma = lambda techo: float((techo - s[mejor]) / (techo - s_base))
    res["techo"] = dict(mejor=mejor, rmse_mejor=s[mejor], rmse_persistencia=s_base,
                        conjunto=dict(modelos=DESDE_CERO + CLASICOS, rmse=s_conj, gamma=gamma(s_conj)),
                        oraculo_por_estacion=dict(rmse=s_oraculo, gamma=gamma(s_oraculo)))
    pr = pd.read_csv("resultados/e2/predictibilidad.csv", dtype={"estacion": str}).set_index("estacion")
    rel = (np.sqrt(est[mejor] / n_est) / np.sqrt(est["Persistencia"] / n_est)).rename("rel")
    j = pr.join(rel, how="inner").dropna(subset=["omega", "rel"])
    rho = stats.spearmanr(j.omega, j.rel)
    res["techo"]["predictibilidad"] = dict(n_estaciones=int(len(j)), spearman_omega_vs_rmse_relativo=float(rho[0]),
                                           p=float(rho[1]))

    # potencia
    r_est = pd.read_csv("resultados/e2/estadistica/rmse_por_estacion.csv", index_col=0)
    d = (r_est[mejor] - r_est["Ridge"]).dropna()
    m_fam = 10
    z = stats.norm.ppf(1 - ALFA / (2 * m_fam)) + stats.norm.ppf(0.8)
    are = 3 / np.pi                                                      # eficiencia de Wilcoxon frente a t (normal)
    mde = float(z * d.std(ddof=1) / np.sqrt(len(d) * are))
    finales = []
    for m in DESDE_CERO + PREENTRENADOS:
        f = pd.read_csv(f"resultados/e2/finales/{m}.csv")
        finales.append(f[f.semilla.isin(SEM_E2)][["modelo", "origen", "semilla", "rmse"]])
    F = pd.concat(finales)
    de_sem = F.groupby(["modelo", "origen"]).rmse.std().mean()
    res["potencia"] = dict(contraste=f"{mejor} - Ridge", n_estaciones=int(len(d)), de_diferencias=float(d.std(ddof=1)),
                           diferencia_media=float(d.mean()), m_holm=m_fam, potencia=0.8,
                           diferencia_minima_detectable=mde, delta=0.02,
                           de_entre_semillas_media=float(de_sem),
                           error_estandar_media_6_semillas=float(de_sem / np.sqrt(len(SEM_E2))))

    # componentes de varianza
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    a = anova_lm(smf.ols("rmse ~ C(modelo) * C(origen)", F).fit(), typ=2)
    a["proporcion"] = a.sum_sq / a.sum_sq.sum()
    res["componentes_varianza"] = {k.replace("C(", "").replace(")", ""): float(v) for k, v in a.proporcion.items()}
    sin_pre = F[~F.modelo.isin(PREENTRENADOS)]
    a2 = anova_lm(smf.ols("rmse ~ C(modelo) * C(origen)", sin_pre).fit(), typ=2)
    res["componentes_varianza_sin_preentrenados"] = {k.replace("C(", "").replace(")", ""): float(v)
                                                     for k, v in (a2.sum_sq / a2.sum_sq.sum()).items()}
    json.dump(res, open("resultados/e2/techo_potencia.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))
