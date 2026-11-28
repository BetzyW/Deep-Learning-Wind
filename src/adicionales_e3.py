"""Condición 4 de R3 (sección 8.4 del enunciado) y evaluación ampliada: PADRE-TCN frente a la familia de contrastes en
los tres conjuntos adicionales (ETTh1, SDWPF y NOAA ISD-Lite Kansas), con las corridas de `e3_adicionales.py`.

* Familia en cada conjunto: los 8 modelos del Top 10 con corridas (Chronos y Moirai sin entrenar no se corrieron en
  los conjuntos adicionales; se declara) y los baselines fuertes Ridge y gradient boosting: 10 contrastes.
* Métrica: RMSE de cada corrida (origen, semilla) en las unidades del conjunto; media por origen y luego entre los 2
  orígenes. Las métricas de conjuntos distintos no se promedian entre sí.
* Pruebas por conjunto (unidad: la corrida pareada por origen y semilla; 20 pares, 6 en STGCN e Informer en SDWPF y
  NOAA por el recorte declarado): t pareado y Wilcoxon de rangos con signo sobre las diferencias de RMSE; frente a los
  baselines deterministas, las diferencias son las de cada corrida de PADRE con el baseline del mismo origen. Holm
  sobre los 10 contrastes de cada conjunto, por prueba; la decisión usa el máximo de los dos p ajustados.
* Entre conjuntos: Friedman con la corrección de Iman–Davenport sobre los rangos de los 11 modelos en los 4 conjuntos
  (panel principal y los tres adicionales).
* Condición 4: PADRE es mejor en la métrica media que todos los rivales en al menos 2 de los 3 conjuntos y no es
  significativamente peor que ninguno en ninguno.
Salida: resultados/e3/adicionales/analisis.json y resultados/e3/adicionales/resumen.csv
Uso: python src/adicionales_e3.py      (ml_env)
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from estadistica_e2 import holm  # noqa: E402

A = "resultados/e3/adicionales"
CONJUNTOS = {"etth1": "ETTh1 (temperatura del aceite)", "sdwpf": "SDWPF (potencia, kW)",
             "noaa_kansas": "NOAA Kansas (viento, m/s)"}
ENTRENABLES = ["Transformer", "KAN", "TCN", "STGCN", "GRU", "Informer", "DLinear", "PatchTST"]
BASELINES = ["Ridge", "Gradient boosting"]
REFERENCIAS = ["MLP", "CNN", "LSTM", "Persistencia"]


def corridas(conj):
    filas = []
    for m in ["PADRE"] + ENTRENABLES + ["MLP", "CNN", "LSTM"]:
        d = pd.read_csv(f"{A}/{conj}/{m}.csv")
        filas.append(d[["origen", "semilla", "rmse"]].assign(modelo=m))
    b = pd.read_csv(f"{A}/{conj}/baselines.csv")
    b = b[b.modelo.isin(BASELINES + ["Persistencia"])][["modelo", "origen", "rmse"]].assign(semilla=-1)
    return pd.concat(filas + [b], ignore_index=True)


def media(d):
    return d.groupby(["modelo", "origen"]).rmse.mean().groupby("modelo").mean()


if __name__ == "__main__":
    res, resumen = {}, []
    for conj, nombre in CONJUNTOS.items():
        d = corridas(conj)
        med = media(d)
        pa = d[d.modelo == "PADRE"].set_index(["origen", "semilla"]).rmse
        filas = []
        for r in ENTRENABLES + BASELINES:
            dr = d[d.modelo == r]
            if r in BASELINES:
                base = dr.set_index("origen").rmse
                dif = (pa - pa.index.get_level_values("origen").map(base).to_numpy()).to_numpy()
            else:
                rv = dr.set_index(["origen", "semilla"]).rmse
                comun = pa.index.intersection(rv.index)
                dif = (pa.loc[comun] - rv.loc[comun]).to_numpy()
            filas.append(dict(rival=r, pares=len(dif), media_rival=float(med[r]),
                              diferencia=float(med["PADRE"] - med[r]),
                              diferencia_relativa=float((med["PADRE"] - med[r]) / med[r]),
                              t_p=float(stats.ttest_1samp(dif, 0).pvalue),
                              wilcoxon_p=float(stats.wilcoxon(dif).pvalue) if np.any(dif != 0) else 1.0))
        t = pd.DataFrame(filas)
        t["t_p_holm"], t["wilcoxon_p_holm"] = holm(t.t_p), holm(t.wilcoxon_p)
        t["p_max_holm"] = t[["t_p_holm", "wilcoxon_p_holm"]].max(axis=1)
        t["padre_mejor"] = t.diferencia < 0
        t["significativo"] = t.p_max_holm < 0.05
        orden = med[["PADRE"] + ENTRENABLES + BASELINES].sort_values()
        res[conj] = dict(
            nombre=nombre, rmse_padre=float(med["PADRE"]), rmse_persistencia=float(med["Persistencia"]),
            rango_padre=int(list(orden.index).index("PADRE") + 1), de=len(orden),
            mejor=orden.index[0], rmse_mejor=float(orden.iloc[0]),
            mejor_que_todos=bool(t.padre_mejor.all()),
            peor_significativo=t[(~t.padre_mejor) & t.significativo].rival.tolist(),
            mejor_significativo=t[t.padre_mejor & t.significativo].rival.tolist(),
            contrastes=t.to_dict(orient="records"),
            ranking={m: float(v) for m, v in orden.items()})
        for _, r in t.iterrows():
            resumen.append(dict(conjunto=conj, **r.to_dict()))
        print(f"\n== {nombre}: PADRE {med['PADRE']:.4f} (rango {res[conj]['rango_padre']} de {len(orden)}); "
              f"mejor {orden.index[0]} {orden.iloc[0]:.4f}; persistencia {med['Persistencia']:.4f}")
        print(t[["rival", "pares", "media_rival", "diferencia", "diferencia_relativa", "t_p_holm", "wilcoxon_p_holm",
                 "significativo"]].round(4).to_string(index=False))

    # Friedman con Iman–Davenport sobre los 4 conjuntos (panel principal + 3 adicionales)
    modelos = ["PADRE"] + ENTRENABLES + BASELINES
    princ = pd.read_csv("resultados/e3/estadistica/resumen_modelos.csv").set_index("modelo").rmse
    princ.index = princ.index.str.replace("_", " ")
    M = pd.DataFrame({"panel IDEAM": princ.reindex(modelos)} |
                     {c: pd.Series(res[c]["ranking"]).reindex(modelos) for c in CONJUNTOS})
    R = M.rank(axis=0)
    N, k = M.shape[1], M.shape[0]
    chi2 = 12 * N / (k * (k + 1)) * ((R.mean(axis=1) ** 2).sum() - k * (k + 1) ** 2 / 4)
    F_id = (N - 1) * chi2 / (N * (k - 1) - chi2)
    p_id = float(stats.f.sf(F_id, k - 1, (k - 1) * (N - 1)))
    cd = float(stats.studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * N)))
    res["friedman"] = dict(conjuntos=list(M.columns), chi2=float(chi2), iman_davenport_F=float(F_id), p=p_id,
                           rangos_medios=R.mean(axis=1).sort_values().round(3).to_dict(), diferencia_critica_nemenyi=cd)
    mejores = sum(res[c]["mejor_que_todos"] for c in CONJUNTOS)
    peor = {c: res[c]["peor_significativo"] for c in CONJUNTOS}
    res["condicion_4"] = dict(conjuntos_donde_es_mejor_que_todos=mejores,
                              peor_significativo_por_conjunto=peor,
                              cumple=bool(mejores >= 2 and not any(peor.values())))
    print("\nRangos medios (1 = mejor):", res["friedman"]["rangos_medios"])
    print(f"Friedman χ² = {chi2:.2f}; Iman–Davenport F = {F_id:.2f}, p = {p_id:.4f}; DC de Nemenyi = {cd:.2f}")
    print("Condición 4:", res["condicion_4"])
    json.dump(res, open(f"{A}/analisis.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    pd.DataFrame(resumen).to_csv(f"{A}/resumen.csv", index=False)
