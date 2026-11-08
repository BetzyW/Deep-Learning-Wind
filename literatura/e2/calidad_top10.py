"""Valoración de calidad metodológica (sección 3.1) de los estudios de viento que respaldan cada modelo del Top 10.

Uso (desde literatura/e2, después de extraccion.py): python calidad_top10.py → calidad_top10.csv
Un estudio respalda un modelo si es elegible y nombra la arquitectura en el TÍTULO (es su modelo propuesto o
protagonista). Riesgo de fuga «alto» si descompone la señal (EMD, VMD, ondículas…) sin describir la partición;
«moderado» si descompone y describe la partición; «bajo» si no descompone. Repeticiones, equidad de los baselines
(comparación con la persistencia) y pruebas estadísticas salen de la hoja de extracción.
"""
import re
import pandas as pd

from tamizaje_e2 import ARQ

TOP10 = {"Transformer": "Transformer", "KAN": "KAN", "TCN": "TCN", "STGCN": "GNN/GCN", "Chronos": "Chronos",
         "Mamba": "Mamba/SSM", "GRU": "GRU", "Informer": "Informer", "DLinear": "DLinear", "Moirai": "Moirai"}

if __name__ == "__main__":
    h = pd.read_csv("hoja_extraccion.csv")
    filas, resumen = [], []
    for modelo, clave in TOP10.items():
        rx = ARQ[clave][1]
        sub = h[h.titulo.str.contains(rx, case=False, regex=True, na=False)]
        for r in sub.itertuples():
            riesgo = "bajo" if not r.descomposicion else ("moderado" if r.particion_descrita else "alto")
            filas.append(dict(modelo=modelo, anio=r.anio, titulo=r.titulo, doi=r.doi, citas=r.citas,
                              riesgo_fuga=riesgo, particion=r.particion_descrita, repeticiones=r.semillas_o_repeticiones,
                              compara_persistencia=r.compara_persistencia, pruebas=r.pruebas_estadisticas,
                              codigo=r.codigo_publico))
        s = pd.DataFrame([f for f in filas if f["modelo"] == modelo])
        resumen.append(dict(modelo=modelo, estudios=len(s),
                            riesgo_alto=int((s.riesgo_fuga == "alto").sum()) if len(s) else 0,
                            con_particion=int(s.particion.sum()) if len(s) else 0,
                            con_repeticiones=int(s.repeticiones.sum()) if len(s) else 0,
                            con_persistencia=int(s.compara_persistencia.sum()) if len(s) else 0,
                            con_pruebas=int(s.pruebas.sum()) if len(s) else 0,
                            con_codigo=int(s.codigo.sum()) if len(s) else 0))
    pd.DataFrame(filas).to_csv("calidad_top10.csv", index=False)
    r = pd.DataFrame(resumen)
    r.to_csv("calidad_top10_resumen.csv", index=False)
    print(r.to_string(index=False))
