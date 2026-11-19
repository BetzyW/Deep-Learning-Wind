"""Tamizaje PRISMA del E2 sobre las cadenas del problema (C1–C6) y conteo de arquitecturas en los estudios elegibles.

Uso (desde literatura/e2): python tamizaje_e2.py
Entradas: scopus_C.csv, wos_C.txt (exportaciones), s2_C*.json y arxiv_C*.json (de buscar_apis.py).
Salidas: registros_e2.csv (cada registro único con su decisión y las arquitecturas que menciona),
         conteos_prisma_e2.json y frecuencia_arquitecturas.csv.

Criterios (fijados antes de mirar los conteos):
  T1 (título): pronostica la velocidad del viento atmosférico (wind speed + forecast/predict; sin viento solar).
  T2 (resumen): hay resumen y menciona al menos una arquitectura del diccionario ARQ.
  E  (elegibilidad, reglas): artículo de revista o congreso (no revisión ni editorial) y el resumen reporta al menos
     una métrica cuantitativa (RMSE, MAE, MAPE, R2, nMAE…) o una mejora porcentual.
"""
import glob, json, re, collections
import pandas as pd

# Arquitectura -> (familia, expresión regular sobre título + resumen + palabras clave)
ARQ = {
    "Transformer": ("Atención", r"\btransformers?\b(?!.*\b(power|electrical)\s+transformer)"),
    "Informer": ("Atención", r"\binformer\b"),
    "Autoformer": ("Atención", r"\bautoformer\b"),
    "FEDformer": ("Atención", r"\bfedformer\b"),
    "PatchTST": ("Atención", r"\bpatchtst\b|patch time series transformer"),
    "iTransformer": ("Atención", r"\bitransformer\b|inverted transformer"),
    "TFT": ("Atención", r"temporal fusion transformer|\bTFT\b"),
    "Atención (mecanismo)": ("Atención", r"attention mechanism|self-attention|multi-head attention|\battention\b"),
    "TCN": ("Convolucional", r"temporal convolution(al)? network|\bTCN\b"),
    "WaveNet": ("Convolucional", r"\bwavenet\b"),
    "TimesNet": ("Convolucional", r"\btimesnet\b"),
    "CNN": ("Convolucional", r"convolutional neural network|\bCNN\b|\b1D-?CNN\b"),
    "N-BEATS": ("MLP profundo", r"\bN-?BEATS\b"),
    "N-HiTS": ("MLP profundo", r"\bN-?HiTS\b"),
    "DLinear": ("MLP profundo", r"\bD-?Linear\b|\bNLinear\b"),
    "TSMixer": ("MLP profundo", r"\bTSMixer\b"),
    "LSTM/BiLSTM": ("Recurrente", r"\bLSTM\b|long short-term memory|\bBiLSTM\b|bidirectional LSTM"),
    "GRU": ("Recurrente", r"\bGRU\b|gated recurrent unit"),
    "xLSTM": ("Recurrente", r"\bxLSTM\b"),
    "DeepAR": ("Probabilístico", r"\bDeepAR\b"),
    "Mamba/SSM": ("Espacio de estados", r"\bmamba\b|selective state space|structured state space|\bS4\b"),
    "KAN": ("Espacio de estados", r"kolmogorov-arnold|\bKAN\b"),
    "GNN/GCN": ("Grafos", r"graph neural network|graph convolution|\bGCN\b|\bGNN\b|graph attention|\bGAT\b|spatio-?temporal graph"),
    "Chronos": ("Preentrenado", r"\bchronos\b"),
    "TimesFM": ("Preentrenado", r"\btimesfm\b"),
    "Moirai": ("Preentrenado", r"\bmoirai\b"),
    "Lag-Llama": ("Preentrenado", r"lag-?llama"),
    "TimeGPT": ("Preentrenado", r"\btimegpt\b"),
    "LLM (adaptado)": ("Preentrenado", r"large language model|\bLLM\b|\bGPT-?2\b|time-?llm"),
    "Difusión": ("Probabilístico", r"diffusion model|denoising diffusion|\bDDPM\b"),
    "Flujos normalizantes": ("Probabilístico", r"normali[sz]ing flow"),
    "Transferencia (fine-tuning)": ("Preentrenado", r"transfer learning|fine-?tun"),
}
METRICA = r"\b(n?RMSE|n?MAE|MAPE|sMAPE|MASE|CRPS|R\s?\^?2|R-squared|coefficient of determination)\b|\d+(\.\d+)?\s?%"
NO_PRIMARIO = r"\b(review|survey|editorial|erratum|retracted|correction)\b"


def norm(t):
    return re.sub(r"[^a-z0-9]", "", str(t).lower())


def doi_norm(d):
    return str(d).lower().strip() if isinstance(d, str) and d.strip() else None


def cargar():
    recs = []
    sc = pd.read_csv("scopus_C.csv", dtype=str)
    for r in sc.to_dict("records"):
        ab = r.get("Abstract") or ""
        recs.append(dict(fuente="Scopus", titulo=r["Title"], anio=int(r["Year"]),
                         resumen="" if ab.startswith("[No abstract") else ab, doi=doi_norm(r.get("DOI")),
                         citas=int(r.get("Cited by") or 0), venue=r.get("Source title"),
                         tipo=r.get("Document Type") or "", claves=f'{r.get("Author Keywords") or ""}; {r.get("Index Keywords") or ""}'))
    wos = pd.read_csv("wos_C.txt", sep="\t", quoting=3, dtype=str, encoding="utf-8-sig", index_col=False)
    for r in wos.to_dict("records"):
        recs.append(dict(fuente="Web of Science", titulo=r["TI"], anio=int(r["PY"]) if isinstance(r["PY"], str) else None,
                         resumen=r.get("AB") if isinstance(r.get("AB"), str) else "", doi=doi_norm(r.get("DI")),
                         citas=int(r.get("TC") or 0), venue=r.get("SO"), tipo=r.get("DT") or "",
                         claves=f'{r.get("DE") or ""}; {r.get("ID") or ""}'))
    vistos = {}
    for f in sorted(glob.glob("s2_C*.json")):
        for p in json.load(open(f, encoding="utf-8")):
            vistos[p["paperId"]] = p
    for p in vistos.values():
        recs.append(dict(fuente="Semantic Scholar", titulo=p["title"], anio=p.get("year"), resumen=p.get("abstract") or "",
                         doi=doi_norm((p.get("externalIds") or {}).get("DOI")), citas=p.get("citationCount") or 0,
                         venue=p.get("venue"), tipo=" ".join(p.get("publicationTypes") or []), claves=""))
    vistos = {}
    for f in sorted(glob.glob("arxiv_C*.json")):
        for a in json.load(open(f, encoding="utf-8")):
            vistos[a["id"]] = a
    for a in vistos.values():
        recs.append(dict(fuente="arXiv", titulo=a["titulo"], anio=a["anio"], resumen=a["resumen"], doi=None, citas=0,
                         venue="arXiv", tipo="Preprint", claves=""))
    return recs


def deduplicar(recs):
    """Mismo DOI o mismo título normalizado; se conserva el primero (Scopus, WoS, S2, arXiv) y se completan campos."""
    unicos, por_clave = [], {}
    for r in recs:
        claves = [k for k in (r["doi"], norm(r["titulo"])) if k]
        previo = next((por_clave[k] for k in claves if k in por_clave), None)
        if previo is None:
            r["fuentes"] = [r["fuente"]]
            unicos.append(r)
            previo = r
        else:
            previo["fuentes"].append(r["fuente"])
            for c in ("resumen", "doi", "venue", "claves"):
                previo[c] = previo[c] or r[c]
            previo["citas"] = max(previo["citas"] or 0, r["citas"] or 0)
        for k in claves:
            por_clave.setdefault(k, previo)
    return unicos


if __name__ == "__main__":
    recs = cargar()
    n_id = collections.Counter(r["fuente"] for r in recs)
    u = deduplicar(recs)
    for r in u:
        t = str(r["titulo"])
        texto = f'{t} {r["resumen"]} {r["claves"]}'
        r["t1"] = bool(re.search(r"wind speed", t, re.I) and re.search(r"forecast|predict", t, re.I)
                       and not re.search(r"solar wind|interplanetary|coronal", texto, re.I)
                       and 2018 <= (r["anio"] or 0) <= 2026)
        r["arquitecturas"] = [a for a, (_, rx) in ARQ.items() if re.search(rx, texto, re.I)]
        r["t2"] = r["t1"] and bool(r["resumen"]) and bool(r["arquitecturas"])
        r["no_primario"] = bool(re.search(NO_PRIMARIO, f'{t} {r["tipo"]}', re.I))
        r["metrica"] = bool(re.search(METRICA, r["resumen"]))
        r["elegible"] = r["t2"] and not r["no_primario"] and r["metrica"]
        r["citas_por_anio"] = (r["citas"] or 0) / max(1, 2026 - (r["anio"] or 2026) + 1)
    p1 = [r for r in u if r["t1"]]
    p2 = [r for r in p1 if r["t2"]]
    el = [r for r in p2 if r["elegible"]]
    conteos = dict(identificados=dict(n_id), total_identificados=len(recs), duplicados=len(recs) - len(u),
                   unicos=len(u), excl_titulo_o_periodo=len(u) - len(p1),
                   sin_resumen=sum(1 for r in p1 if not r["resumen"]),
                   sin_arquitectura=sum(1 for r in p1 if r["resumen"] and not r["arquitecturas"]),
                   tamizados_resumen=len(p1), evaluados_elegibilidad=len(p2),
                   excl_revision_o_no_primario=sum(1 for r in p2 if r["no_primario"]),
                   excl_sin_metrica=sum(1 for r in p2 if not r["no_primario"] and not r["metrica"]),
                   elegibles=len(el))
    json.dump(conteos, open("conteos_prisma_e2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    cols = ["titulo", "anio", "fuentes", "doi", "venue", "tipo", "citas", "citas_por_anio", "t1", "t2", "no_primario",
            "metrica", "elegible", "arquitecturas"]
    df = pd.DataFrame(u)[cols]
    df["fuentes"] = df.fuentes.str.join("; ")
    df["arquitecturas"] = df.arquitecturas.str.join("; ")
    df.sort_values(["elegible", "citas"], ascending=False).to_csv("registros_e2.csv", index=False)
    frec = []
    for a, (fam, _) in ARQ.items():
        est = [r for r in el if a in r["arquitecturas"]]
        en_titulo = [r for r in est if re.search(ARQ[a][1], str(r["titulo"]), re.I)]
        frec.append(dict(arquitectura=a, familia=fam, estudios=len(est), en_titulo=len(en_titulo),
                         estudios_2023_2026=sum(1 for r in est if (r["anio"] or 0) >= 2023),
                         citas_por_anio_mediana=float(pd.Series([r["citas_por_anio"] for r in est]).median()) if est else 0))
    frec = pd.DataFrame(frec).sort_values("estudios", ascending=False)
    frec.to_csv("frecuencia_arquitecturas.csv", index=False)
    print(json.dumps(conteos, ensure_ascii=False, indent=1))
    print(frec.to_string(index=False))
