"""Hoja de extracción estructurada (sección 3.1) de los estudios elegibles, con campos fijos leídos del resumen.

Uso (desde literatura/e2, después de tamizaje_e2.py): python extraccion.py
Salidas: hoja_extraccion.csv (un estudio por fila) y resumen_extraccion.json (proporciones).
Cada campo es una regla explícita sobre título + resumen; «no» significa «el resumen no lo menciona», no que el
estudio no lo haga. Los estudios que respaldan cada modelo del Top 10 se revisan a mano después (columna revisado).
"""
import json, re
import pandas as pd

from tamizaje_e2 import cargar, deduplicar

CAMPOS = {
    "particion_descrita": r"training (set|data|period)|test(ing)? (set|data|period)|validation (set|data)|split|"
                          r"cross-validation|rolling|walk-forward|hold-?out|\d+\s?%\s?(for|of the data)",
    "semillas_o_repeticiones": r"\bseeds?\b|independent runs|repeated (runs|experiments|trials)|\b\d+ runs\b|"
                               r"standard deviation of|mean (and|±) (std|standard)",
    "pruebas_estadisticas": r"diebold|mariano|\bDM test|wilcoxon|friedman|nemenyi|t-test|model confidence set|"
                            r"giacomini|clark-west|statistically significant|significance test",
    "codigo_publico": r"github|code (is|are) (publicly )?available|open-?source code|publicly available code",
    "datos_publicos": r"publicly available (data|dataset)|open (data|dataset)|NREL|ERA5|NOAA|NASA|kaggle",
    "descomposicion": r"\bEMD\b|\bEEMD\b|CEEMDAN|\bVMD\b|ICEEMDAN|wavelet|empirical mode|variational mode|"
                      r"singular spectrum|\bSSA\b|decomposition",
    "compara_persistencia": r"persistence|naive|na[iï]ve",
    "multiestacion": r"multi-?(site|station|location)|spatio-?temporal|neighbo(u)?ring (sites|stations)|"
                     r"several (sites|stations)|\bgraph\b",
    "horizonte_24h_o_mas": r"24[- ]?h|day-ahead|24 hours|one day ahead|48[- ]?h|72[- ]?h",
    "incertidumbre": r"probabilistic|interval|quantile|uncertainty|CRPS|pinball",
}
METRICAS = ["RMSE", "MAE", "MAPE", "sMAPE", "MASE", "R2", "nRMSE", "CRPS", "pinball", "MSE"]
RX_METRICA = {m: rf"\b{m}\b" if m != "R2" else r"\bR\s?\^?2\b|R-squared|coefficient of determination"
              for m in METRICAS}

if __name__ == "__main__":
    reg = pd.read_csv("registros_e2.csv")
    elegibles = set(reg.loc[reg.elegible, "titulo"].map(lambda t: re.sub(r"[^a-z0-9]", "", str(t).lower())))
    u = deduplicar(cargar())
    filas = []
    for r in u:
        clave = re.sub(r"[^a-z0-9]", "", str(r["titulo"]).lower())
        if clave not in elegibles:
            continue
        texto = f'{r["titulo"]} {r["resumen"]}'
        f = dict(titulo=r["titulo"], anio=r["anio"], doi=r["doi"], venue=r["venue"], citas=r["citas"],
                 fuentes="; ".join(r["fuentes"]))
        fila_reg = reg[reg.titulo == r["titulo"]]
        f["arquitecturas"] = fila_reg.arquitecturas.iloc[0] if len(fila_reg) else ""
        f["metricas"] = "; ".join(m for m, rx in RX_METRICA.items() if re.search(rx, texto, re.I))
        for c, rx in CAMPOS.items():
            f[c] = bool(re.search(rx, texto, re.I))
        f["revisado"] = False
        filas.append(f)
    h = pd.DataFrame(filas).sort_values("citas", ascending=False)
    h.to_csv("hoja_extraccion.csv", index=False)
    resumen = {c: dict(n=int(h[c].sum()), pct=round(100 * h[c].mean(), 1)) for c in CAMPOS}
    resumen["n_estudios"] = len(h)
    # cruces que sirven para el gap
    resumen["persistencia_y_pruebas"] = int((h.compara_persistencia & h.pruebas_estadisticas).sum())
    resumen["multiestacion_y_24h"] = int((h.multiestacion & h.horizonte_24h_o_mas).sum())
    resumen["preentrenado_y_persistencia_y_pruebas"] = int(
        (h.arquitecturas.fillna("").str.contains("Chronos|Moirai|TimesFM|Lag-Llama|TimeGPT|LLM")
         & h.compara_persistencia & h.pruebas_estadisticas).sum())
    resumen["semillas_y_pruebas"] = int((h.semillas_o_repeticiones & h.pruebas_estadisticas).sum())
    json.dump(resumen, open("resumen_extraccion.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(resumen, ensure_ascii=False, indent=1))
