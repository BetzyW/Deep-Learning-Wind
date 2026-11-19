"""Análisis del espacio de arquitecturas (secciones 14.6 y 14.7 del enunciado) para los modelos con búsqueda:
MLP, CNN, GRU, LSTM, TCN, KAN, Transformer (redes desde cero) y los cabezales de Chronos y Moirai.

Por modelo, sobre las arquitecturas de la fase 1 que entraron a la selección (las 1 000, o la muestra de 200 de las
enmiendas 2 y 3):
* Modelo lineal de efectos principales e interacciones de dropout (fórmula de la sección 14.6), con IC 95 % (errores
  HC3), ANOVA tipo II por factor con Holm sobre los factores, y proporción de la suma de cuadrados de cada factor.
* Tendencia en el ancho: regresión sobre log2 w y (log2 w)² por capa (rendimientos decrecientes si el término
  cuadrático es positivo y significativo).
* Sin dropout frente a con dropout: Wilcoxon pareado por combinación de anchos (bloques con ambos tipos evaluados).
* Interacción ancho × dropout: término log2(ancho total) × (algún dropout).
* Frente de Pareto (RMSE de validación frente a parámetros y frente a segundos).
* Sensibilidad: desviación entre arquitecturas (fase 1) frente a desviación entre semillas (fase 2).
* Sobreajuste: razón RMSE validación / entrenamiento por (tercil de ancho total, con o sin dropout).
* Fase 2: correlación de Spearman entre fases en los finalistas, ganador con la regla declarada, finalistas
  equivalentes al mejor (TOST pareado por semilla con ±δ) y los que caen dentro de δ/2; posición del ganador (borde
  del espacio); optimismo de la fase 1 (RMSE de fase 1 del ganador frente a su media en fase 2).
* Costo: segundos de GPU de ajuste previo, fase 1 y fase 2.
Salida: resultados/e2/busqueda_analisis/ (CSV, resumen.json y un mapa de calor por modelo).
Uso: python src/analisis_busqueda.py      (ml_env)
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from estadistica_e2 import DELTA, holm, tost  # noqa: E402

CARPETA = "resultados/e2/busqueda"
SALIDA = "resultados/e2/busqueda_analisis"
MODELOS = ["MLP", "CNN", "GRU", "LSTM", "TCN", "KAN", "Transformer", "Chronos", "Moirai"]
ANCHOS = [16, 32, 64, 128, 256]


def fase1(m):
    d = pd.read_csv(f"{CARPETA}/{m}_fase1.csv")
    d = d[d.estado == "ok"].drop_duplicates("id", keep="last")
    ruta = f"{CARPETA}/{m}_muestra_fase1.json"
    if os.path.exists(ruta):
        d = d[d.id.isin(json.load(open(ruta)))]
    d = d.copy()
    d["algun_dropout"] = (d[["b1", "b2", "b3"]].sum(1) > 0).astype(int)
    d["log2_total"] = np.log2(d[["w1", "w2", "w3"]].sum(1))
    return d


def efectos(d):
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    f = "rmse_val ~ C(w1) + C(w2) + C(w3) + b1 + b2 + b3 + b1:b2 + b1:b3 + b2:b3"
    r = smf.ols(f, d).fit()
    rc = r.get_robustcov_results("HC3")
    ic = rc.conf_int()
    coef = pd.DataFrame(dict(termino=r.params.index, coef=rc.params, ic_inf=ic[:, 0], ic_sup=ic[:, 1], p=rc.pvalues))
    a = anova_lm(r, typ=2)
    a["prop_ss"] = a["sum_sq"] / a["sum_sq"].sum()
    a = a.drop(index="Residual").reset_index().rename(columns={"index": "factor", "PR(>F)": "p"})
    a["p_holm"] = holm(a.p)
    tend = smf.ols("rmse_val ~ np.log2(w1) + I(np.log2(w1)**2) + np.log2(w2) + I(np.log2(w2)**2)"
                   " + np.log2(w3) + I(np.log2(w3)**2)", d).fit(cov_type="HC3")
    inter = smf.ols("rmse_val ~ log2_total * algun_dropout", d).fit(cov_type="HC3")
    return coef, a, dict(r2=float(r.rsquared),
                         tendencia={k: dict(coef=float(tend.params[k]), p=float(tend.pvalues[k]))
                                    for k in tend.params.index if k != "Intercept"},
                         ancho_x_dropout=dict(coef=float(inter.params["log2_total:algun_dropout"]),
                                              p=float(inter.pvalues["log2_total:algun_dropout"])))


def sin_vs_con_dropout(d):
    g = d.assign(c=d.w1.astype(str) + "-" + d.w2.astype(str) + "-" + d.w3.astype(str))
    sin = g[g.algun_dropout == 0].groupby("c").rmse_val.mean()
    con = g[g.algun_dropout == 1].groupby("c").rmse_val.mean()
    comun = sin.index.intersection(con.index)
    if len(comun) < 6:
        return dict(n_bloques=int(len(comun)), nota="pocos bloques con ambos tipos (muestra de 200)")
    dif = (con.loc[comun] - sin.loc[comun]).to_numpy()
    w = stats.wilcoxon(dif)
    rangos = stats.rankdata(np.abs(dif))
    return dict(n_bloques=int(len(comun)), diferencia_mediana_con_menos_sin=float(np.median(dif)), p=float(w.pvalue),
                biserial=float((rangos[dif > 0].sum() - rangos[dif < 0].sum()) / rangos.sum()))


def pareto(d, x):
    d = d.sort_values([x, "rmse_val"])
    mejor, ids = np.inf, []
    for _, r in d.iterrows():
        if r.rmse_val < mejor:
            mejor = r.rmse_val
            ids.append(r.id)
    return ids


def mapa(d, m, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pats = [f"{a}{b}{c}" for a in "01" for b in "01" for c in "01"]
    fig, ejes = plt.subplots(8, 1, figsize=(11, 13), sharex=True)
    vmin, vmax = d.rmse_val.quantile(0.02), d.rmse_val.quantile(0.98)
    for ax, pat in zip(ejes, pats):
        sub = d[(d.b1.astype(str) + d.b2.astype(str) + d.b3.astype(str)) == pat]
        M = np.full((5, 25), np.nan)
        for _, r in sub.iterrows():
            M[ANCHOS.index(r.w1), ANCHOS.index(r.w2) * 5 + ANCHOS.index(r.w3)] = r.rmse_val
        im = ax.imshow(M, aspect="auto", cmap="viridis_r", vmin=vmin, vmax=vmax)
        ax.set_yticks(range(5), ANCHOS, fontsize=7)
        ax.set_ylabel(f"b={pat}\nw1", fontsize=8)
    ejes[-1].set_xticks(range(25), [f"{a}/{b}" for a in ANCHOS for b in ANCHOS], rotation=90, fontsize=7)
    ejes[-1].set_xlabel("w2 / w3")
    fig.colorbar(im, ax=ejes, shrink=0.6, label="RMSE de validación (m/s), fase 1")
    fig.suptitle(f"{m}: superficie de la fase 1 (gris = no evaluada)")
    fig.savefig(ruta, dpi=110, bbox_inches="tight")
    plt.close(fig)


def fase2(m, d1):
    d = pd.read_csv(f"{CARPETA}/{m}_fase2.csv")
    d = d[d.estado == "ok"]
    r = d.groupby("id").agg(rmse=("rmse_val", "mean"), de=("rmse_val", "std"), n=("rmse_val", "size"),
                            parametros=("parametros", "first"), referencia=("referencia", "max")).sort_values("rmse")
    r = r[r.n >= 5]
    empatadas = r[r.rmse < r.rmse.iloc[0] + DELTA / 2]
    ganador = empatadas.sort_values("parametros").index[0]
    piv = d.pivot_table(index="semilla", columns="id", values="rmse_val")
    mejor = r.index[0]
    equiv = [i for i in r.index if i == mejor or tost((piv[i] - piv[mejor]).dropna().to_numpy())["equivalentes"]]
    f1 = d1.set_index("id").rmse_val
    fin = r[r.referencia == 0].index.intersection(f1.index)
    rho = stats.spearmanr(f1.loc[fin], r.loc[fin, "rmse"]) if len(fin) >= 4 else None
    w = [int(x) for x in ganador[1:].split("_d")[0].split("-")]
    return dict(ganador=ganador, mejor_media=mejor, empatados_delta_medio=list(empatadas.index),
                equivalentes_tost=equiv, n_finalistas=int(len(r)),
                spearman_fase1_fase2=None if rho is None else dict(rho=float(rho[0]), p=float(rho[1]), n=len(fin)),
                de_entre_semillas_media=float(r.de.mean()),
                ganador_en_borde=bool(any(x in (16, 256) for x in w)), ganador_todo_extremo=bool(len(set(w)) == 1
                                                                                         and w[0] in (16, 256)),
                ganador_rmse_fase1=float(f1.get(ganador, np.nan)), ganador_rmse_fase2=float(r.loc[ganador, "rmse"]),
                optimismo_fase1=float(r.loc[ganador, "rmse"] - f1.get(ganador, np.nan))), r


if __name__ == "__main__":
    os.makedirs(SALIDA, exist_ok=True)
    resumen, costo = {}, []
    for m in MODELOS:
        d1 = fase1(m)
        coef, anova, extra = efectos(d1)
        coef.to_csv(f"{SALIDA}/{m}_coeficientes.csv", index=False)
        anova.to_csv(f"{SALIDA}/{m}_anova.csv", index=False)
        sens_arq = float(d1.rmse_val.std())
        f2, tabla2 = fase2(m, d1)
        tabla2.to_csv(f"{SALIDA}/{m}_fase2_resumen.csv")
        d1["tercil_ancho"] = pd.qcut(d1.log2_total, 3, labels=["angosta", "media", "ancha"]).astype(str)
        brecha = (d1.assign(razon=d1.rmse_val / d1.rmse_train).groupby(["tercil_ancho", "algun_dropout"]).razon
                  .median().round(3))
        resumen[m] = dict(n_arquitecturas=int(len(d1)), rmse_min=float(d1.rmse_val.min()),
                          rmse_rango=float(d1.rmse_val.max() - d1.rmse_val.min()), de_entre_arquitecturas=sens_arq,
                          razon_arquitecturas_semillas=float(sens_arq / f2["de_entre_semillas_media"]),
                          anova={r.factor: dict(prop_ss=float(r.prop_ss), p_holm=float(r.p_holm))
                                 for r in anova.itertuples()},
                          sin_vs_con_dropout=sin_vs_con_dropout(d1), pareto_parametros=pareto(d1, "parametros"),
                          pareto_segundos=pareto(d1, "segundos"),
                          razon_val_train={f"{a}|{b}": float(v) for (a, b), v in brecha.items()}, **extra, fase2=f2)
        mapa(d1, m, f"{SALIDA}/{m}_mapa_fase1.png")
        hp = pd.read_csv(f"{CARPETA}/{m}_hp.csv")
        f2d = pd.read_csv(f"{CARPETA}/{m}_fase2.csv")
        f1d = pd.read_csv(f"{CARPETA}/{m}_fase1.csv")
        costo.append(dict(modelo=m, horas_ajuste_previo=hp.segundos.sum() / 3600 if "segundos" in hp else np.nan,
                          horas_fase1=f1d.segundos.sum() / 3600, horas_fase2=f2d.segundos.sum() / 3600))
        print(f"{m}: {len(d1)} arquitecturas, R² {extra['r2']:.2f}, ganador {f2['ganador']}, "
              f"ρ fases {f2['spearman_fase1_fase2']}", flush=True)
    c = pd.DataFrame(costo)
    c["horas_total"] = c[["horas_ajuste_previo", "horas_fase1", "horas_fase2"]].sum(1)
    c.to_csv(f"{SALIDA}/costo.csv", index=False)
    resumen["costo_total_horas"] = float(c.horas_total.sum())
    json.dump(resumen, open(f"{SALIDA}/resumen.json", "w"), indent=1, ensure_ascii=False)
    print(c.round(2).to_string(index=False))
