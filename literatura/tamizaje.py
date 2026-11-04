import json, re, collections
s2 = {p["paperId"]: p for p in json.load(open("s2.json", encoding="utf-8"))}.values()
ax = json.load(open("arxiv.json", encoding="utf-8"))
recs = [dict(fuente="Semantic Scholar", titulo=p["title"], anio=p["year"], resumen=p.get("abstract") or "",
             doi=(p.get("externalIds") or {}).get("DOI"), citas=p.get("citationCount", 0), venue=p.get("venue"),
             autores=[a["name"] for a in (p.get("authors") or [])][:3], tipos=p.get("publicationTypes")) for p in s2]
recs += [dict(fuente="arXiv", titulo=a["titulo"], anio=a["anio"], resumen=a["resumen"], doi=None, citas=None,
              venue="arXiv", autores=a["autores"], url=a["id"]) for a in ax]
n_id = collections.Counter(r["fuente"] for r in recs)
norm = lambda t: re.sub(r"[^a-z0-9]", "", t.lower())
vistos, unicos = set(), []
for r in recs:
    k = norm(r["titulo"])
    if k in vistos: continue
    vistos.add(k); unicos.append(r)
dup = len(recs) - len(unicos)
# Tamizaje 1 (título): debe tratar de pronóstico de velocidad del viento atmosférica
T = lambda r: re.search(r"wind speed", r["titulo"], re.I) and re.search(r"forecast|predict", r["titulo"], re.I) \
    and not re.search(r"solar wind|interplanetary|coronal", r["titulo"] + r["resumen"], re.I)
p1 = [r for r in unicos if T(r)]
# Tamizaje 2 (resumen): disponible y menciona al menos dos familias base de redes como modelos evaluados
fam = {"MLP": r"multi-?layer perceptron|\bMLP\b|\bANN\b|feed-?forward|back ?propagation neural|\bBPNN?\b|artificial neural network",
       "RNN": r"\bLSTM|long short-term|\bGRU\b|gated recurrent|recurrent neural|\bRNN\b",
       "CNN": r"convolutional|\bCNN\b|\bTCN\b|temporal convolution",
       "Transformer": r"transformer|informer|autoformer|self-attention"}
for r in p1:
    r["familias"] = [f for f, rx in fam.items() if re.search(rx, r["titulo"] + " " + r["resumen"], re.I)]
sin_res = [r for r in p1 if not r["resumen"]]
p2 = [r for r in p1 if r["resumen"] and len(r["familias"]) >= 2]
print(dict(identificados=n_id, duplicados=dup, unicos=len(unicos), excl_titulo=len(unicos) - len(p1),
           sin_resumen=len(sin_res), excl_menos_2_familias=len(p1) - len(sin_res) - len(p2), pasan=len(p2)))
json.dump(p2, open("tamiz2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
print(collections.Counter("+".join(r["familias"]) for r in p2).most_common(10))
