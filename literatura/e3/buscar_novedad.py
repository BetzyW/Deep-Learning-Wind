"""Búsqueda de novedad del modelo original del E3 (sección 6.3 del enunciado) en tres bases abiertas.

Bases: OpenAlex (índice de Microsoft Academic ampliado; cubre Scopus/WoS en buena parte), Semantic Scholar y arXiv.
Scopus y Web of Science exigen credenciales institucionales y se consultan aparte desde el navegador.
Cada cadena combina (i) el mecanismo, (ii) el problema y (iii) sinónimos; hay cadenas del mecanismo en otras
aplicaciones para identificar su origen. Se registra la fecha, el número de resultados y los 50 primeros títulos.
Salida: literatura/e3/busquedas_{fecha}.csv (registro) y literatura/e3/resultados_{fecha}.csv (títulos para revisar);
la fecha en el nombre conserva cada repetición de la búsqueda (la primera fue el 20-nov-2026).
Uso: python literatura/e3/buscar_novedad.py      (desde la raíz del repositorio)
"""
import csv, json, os, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

CARPETA = os.path.dirname(os.path.abspath(__file__))
CADENAS = [  # (id, propósito, texto libre para OpenAlex/Semantic Scholar, consulta arXiv)
    ("C1", "mecanismo + problema", "twin neural network regression wind speed",
     'all:"twin neural network" AND all:wind'),
    ("C2", "mecanismo + problema (sinónimos)", "pairwise difference regression wind speed forecasting",
     'all:"pairwise difference" AND all:wind'),
    ("C3", "mecanismo + problema (siamesa)", "siamese network regression wind speed forecasting",
     'all:siamese AND all:"wind speed"'),
    ("C4", "análogos + redes neuronales + viento", "analog ensemble neural network wind speed forecasting",
     'all:"analog ensemble" AND all:"neural network"'),
    ("C5", "diferencias entre pares + series de tiempo", "twin neural network regression time series forecasting",
     'all:"twin neural network" AND all:forecasting'),
    ("C6", "mecanismo en otras aplicaciones (origen)", "twin neural network regression",
     'all:"twin neural network regression"'),
    ("C7", "PADRE (origen del equipo)", "pairwise difference regression PADRE",
     'all:"pairwise difference regression"'),
    ("C8", "anclas por análogos + aprendizaje profundo", "analog anchors deep learning forecasting difference",
     'all:analog AND all:anchor AND all:forecasting'),
    ("C9", "español", "red neuronal gemela regresión velocidad del viento",
     'all:"red neuronal" AND all:viento'),
]
CLAVES = ["twin", "pairwise", "siamese", "analog", "padre", "difference"]


def obtener(url, intentos=4):
    for k in range(intentos):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "busqueda-academica/1.0"})
            return urllib.request.urlopen(req, timeout=60).read()
        except Exception as e:  # límite de tasa: esperar y repetir
            time.sleep(5 * (k + 1))
            err = e
    raise err


def openalex(texto):
    u = "https://api.openalex.org/works?" + urllib.parse.urlencode(
        {"filter": f"title_and_abstract.search:{texto}", "per-page": 50})
    d = json.loads(obtener(u))
    return d["meta"]["count"], [(w.get("publication_year"), w.get("title") or "", w.get("doi") or w["id"])
                                for w in d["results"]]


def semantic(texto):
    u = "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode(
        {"query": texto, "limit": 50, "fields": "title,year,externalIds"})
    d = json.loads(obtener(u))
    return d.get("total", 0), [(p.get("year"), p.get("title") or "",
                                (p.get("externalIds") or {}).get("DOI") or p.get("paperId"))
                               for p in d.get("data", [])]


def arxiv(consulta):
    u = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": consulta, "max_results": 50})
    r = ET.fromstring(obtener(u))
    ns = {"a": "http://www.w3.org/2005/Atom", "o": "http://a9.com/-/spec/opensearch/1.1/"}
    total = int(r.find("o:totalResults", ns).text)
    return total, [(e.find("a:published", ns).text[:4], " ".join(e.find("a:title", ns).text.split()),
                    e.find("a:id", ns).text) for e in r.findall("a:entry", ns)]


if __name__ == "__main__":
    fecha = time.strftime("%Y-%m-%d")
    reg, res = [], []
    for cid, prop, texto, qa in CADENAS:
        for base, f, q in [("OpenAlex", openalex, texto), ("Semantic Scholar", semantic, texto), ("arXiv", arxiv, qa)]:
            try:
                n, filas = f(q)
            except Exception as e:
                n, filas = -1, []
                print(base, cid, "falló:", e)
            reg.append(dict(fecha=fecha, cadena=cid, proposito=prop, base=base, consulta=q, resultados=n,
                            revisados=len(filas)))
            for anio, titulo, ident in filas:
                t = titulo.lower()
                res.append(dict(cadena=cid, base=base, anio=anio, titulo=titulo, id=ident,
                                candidato=int(any(k in t for k in CLAVES) and ("wind" in t or "viento" in t))))
            print(f"{cid} {base:16s} {n:>7}", flush=True)
            time.sleep(3)
    for nombre, filas in [(f"busquedas_{fecha}.csv", reg), (f"resultados_{fecha}.csv", res)]:
        with open(os.path.join(CARPETA, nombre), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas[0]))
            w.writeheader(); w.writerows(filas)
    print("candidatos:", sum(r["candidato"] for r in res))
