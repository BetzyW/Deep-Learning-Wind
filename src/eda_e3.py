"""EDA específico del E3: la estructura de análogos diarios que motiva las anclas de PADRE-TCN.

Para cada estación (serie horaria de velocidad del panel del IDEAM, solo el periodo de entrenamiento, antes de 2025):
* autocorrelación de la velocidad en los rezagos 1 a 168 h;
* correlación entre el objetivo v(t + 24) y el valor de cada ancla, v(t − 24 (j − 1)) para j = 1, …, 7, frente a la de
  rezagos que no son múltiplos de 24 (t − 12, t − 36), para mostrar que la misma hora de los días anteriores es un
  mejor análogo que las horas intermedias.
* fracción de horas con viento fuerte (≥ 6 m/s), el régimen donde todos los modelos fallan.
Figura: book/figuras/e3_acf_anclas.png (mediana entre estaciones y rango intercuartílico). Salida:
resultados/e3/eda_e3.json
Uso: python src/eda_e3.py      (ml_env)
"""
import json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from origen_movil import leer_config
    cfg = leer_config()
    d = pd.read_csv(cfg["panel"], usecols=["CodigoEstacion", "hora", "vel"], parse_dates=["hora"])
    d = d[(d.hora < "2025-01-01") & ~d.CodigoEstacion.astype(str).isin(cfg["excluir_estaciones"])]
    acf, corr_obj, fuerte = [], [], []
    for est, g in d.groupby("CodigoEstacion"):
        s = g.set_index("hora").vel.asfreq("h")
        if s.notna().sum() < 24 * 120:                                   # al menos 120 días con datos
            continue
        acf.append([s.autocorr(k) for k in range(1, 169)])
        y = s.shift(-24)                                                 # objetivo: v(t + 24)
        corr_obj.append({**{f"ancla {j}": y.corr(s.shift(24 * (j - 1))) for j in range(1, 8)},
                         "t − 12": y.corr(s.shift(12)), "t − 36": y.corr(s.shift(36))})
        fuerte.append(float((s >= 6).mean()))
    A = np.array(acf)
    C = pd.DataFrame(corr_obj)
    res = dict(estaciones=len(A), acf_mediana={int(k): float(np.nanmedian(A[:, k - 1])) for k in
                                                 [1, 12, 24, 36, 48, 72, 96, 120, 144, 168]},
               corr_objetivo_mediana=C.median().round(3).to_dict(),
               corr_objetivo_p25=C.quantile(0.25).round(3).to_dict(),
               corr_objetivo_p75=C.quantile(0.75).round(3).to_dict(),
               fraccion_fuerte_mediana=float(np.median(fuerte)), fraccion_fuerte_max=float(np.max(fuerte)))
    json.dump(res, open("resultados/e3/eda_e3.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    k = np.arange(1, 169)
    ax.fill_between(k, np.nanpercentile(A, 25, 0), np.nanpercentile(A, 75, 0), color="#9fc5e8", alpha=0.6,
                    label="rango intercuartílico entre estaciones")
    ax.plot(k, np.nanmedian(A, 0), color="#0b5394", lw=1.6, label="mediana entre estaciones")
    for j in range(1, 8):
        ax.axvline(24 * j, color="#e69f00", lw=0.8, ls="--", zorder=0)
    ax.set_xlabel("Rezago (horas); las líneas punteadas marcan múltiplos de 24 h (anclas diarias)")
    ax.set_ylabel("Autocorrelación de la velocidad")
    ax.set_title(f"Autocorrelación de la velocidad del viento por estación ({len(A)} estaciones, 2020–2024)")
    ax.set_xlim(1, 168)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig("book/figuras/e3_acf_anclas.png", dpi=130)
    print(json.dumps(res, indent=1, ensure_ascii=False))
