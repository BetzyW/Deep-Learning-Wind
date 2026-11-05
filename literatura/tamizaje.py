"""Tamizaje PRISMA reproducible sobre las cuatro fuentes: Scopus, Web of Science, Semantic Scholar y arXiv.

Uso: python tamizaje.py   (desde la carpeta literatura/)
Entradas: scopus.csv, wos.txt, s2.json, arxiv.json y decisiones_elegibilidad.csv (lectura manual de resúmenes).
Salidas: tamiz2.json (registros que pasan el tamizaje automático, con su decisión de elegibilidad) y conteos PRISMA.
"""
import json, re, collections
import pandas as pd


def norm(t):
    return re.sub(r"[^a-z0-9]", "", str(t).lower())


def doi_norm(d):
    return str(d).lower().strip() if isinstance(d, str) and d.strip() else None


recs = []
sc = pd.read_csv("scopus.csv", dtype=str).rename(columns={"Cited by": "Citas", "Source title": "Revista"})
recs += [dict(fuente="Scopus", titulo=r.Title, anio=int(r.Year), resumen=r.Abstract if isinstance(r.Abstract, str)
              and not r.Abstract.startswith("[No abstract") else "", doi=doi_norm(r.DOI), citas=int(r.Citas or 0),
              venue=r.Revista, autores=str(r.Authors).split("; ")[:3]) for r in sc.itertuples()]
wos = pd.read_csv("wos.txt", sep="\t", quoting=3, dtype=str, encoding="utf-8-sig", index_col=False)
recs += [dict(fuente="Web of Science", titulo=r.TI, anio=int(r.PY) if isinstance(r.PY, str) else None,
              resumen=r.AB if isinstance(r.AB, str) else "", doi=doi_norm(r.DI), citas=int(r.TC or 0), venue=r.SO,
              autores=str(r.AU).split("; ")[:3]) for r in wos.itertuples()]
s2 = {p["paperId"]: p for p in json.load(open("s2.json", encoding="utf-8"))}.values()
recs += [dict(fuente="Semantic Scholar", titulo=p["title"], anio=p["year"], resumen=p.get("abstract") or "",
              doi=doi_norm((p.get("externalIds") or {}).get("DOI")), citas=p.get("citationCount", 0),
              venue=p.get("venue"), autores=[a["name"] for a in (p.get("authors") or [])][:3]) for p in s2]
recs += [dict(fuente="arXiv", titulo=a["titulo"], anio=a["anio"], resumen=a["resumen"], doi=None, citas=0,
              venue="arXiv", autores=a["autores"]) for a in json.load(open("arxiv.json", encoding="utf-8"))]
n_id = collections.Counter(r["fuente"] for r in recs)

# Duplicados: mismo DOI o mismo título normalizado. Se conserva el primero (orden: Scopus, WoS, S2, arXiv) y se
# completan resumen y citas con los de los duplicados.
unicos, por_clave = [], {}
for r in recs:
    claves = [k for k in (r["doi"], norm(r["titulo"])) if k]
    previo = next((por_clave[k] for k in claves if k in por_clave), None)
    if previo is None:
        r["fuentes"] = [r["fuente"]]; unicos.append(r); previo = r
    else:
        previo["fuentes"].append(r["fuente"])
        if not previo["resumen"] and r["resumen"]:
            previo["resumen"] = r["resumen"]
        previo["citas"] = max(previo["citas"] or 0, r["citas"] or 0)
        previo["doi"] = previo["doi"] or r["doi"]
    for k in claves:
        por_clave.setdefault(k, previo)
dup = len(recs) - len(unicos)

# Tamizaje 1 (título): pronóstico de la velocidad del viento atmosférico
T = lambda r: re.search(r"wind speed", str(r["titulo"]), re.I) and re.search(r"forecast|predict", str(r["titulo"]), re.I) \
    and not re.search(r"solar wind|interplanetary|coronal", str(r["titulo"]) + r["resumen"], re.I)
p1 = [r for r in unicos if T(r) and 2018 <= (r["anio"] or 0) <= 2026]
# Tamizaje 2 (resumen): disponible y con al menos dos familias de redes
fam = {"MLP": r"multi-?layer perceptron|\bMLP\b|\bANN\b|feed-?forward|back ?propagation neural|\bBPNN?\b|artificial neural network",
       "RNN": r"\bLSTM|long short-term|\bGRU\b|gated recurrent|recurrent neural|\bRNN\b",
       "CNN": r"convolutional|\bCNN\b|\bTCN\b|temporal convolution",
       "Transformer": r"transformer|informer|autoformer|self-attention"}
for r in p1:
    r["familias"] = [f for f, rx in fam.items() if re.search(rx, str(r["titulo"]) + " " + r["resumen"], re.I)]
sin_res = [r for r in p1 if not r["resumen"]]
p2 = [r for r in p1 if r["resumen"] and len(r["familias"]) >= 2]
p2.sort(key=lambda r: -(r["citas"] or 0))

# Elegibilidad: decisión manual registrada por título normalizado
dec = pd.read_csv("decisiones_elegibilidad.csv", dtype=str)
decision = dict(zip(dec.clave, dec.elegible))
for i, r in enumerate(p2):
    r["orden"], r["clave"] = i, norm(r["titulo"])
    r["elegible"] = decision.get(r["clave"]) == "1"
    r["evaluado"] = r["clave"] in decision
json.dump(p2, open("tamiz2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)

conteos = dict(identificados=dict(n_id), duplicados=dup, unicos=len(unicos), excl_titulo=len(unicos) - len(p1),
               sin_resumen=len(sin_res), excl_menos_2_familias=len(p1) - len(sin_res) - len(p2), tamizados_resumen=len(p1),
               evaluados_elegibilidad=len(p2), sin_decision=sum(not r["evaluado"] for r in p2),
               elegibles=sum(r["elegible"] for r in p2))
json.dump(conteos, open("conteos_prisma.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(conteos)
print(collections.Counter(f for r in p2 if r["elegible"] for f in r["familias"]))
