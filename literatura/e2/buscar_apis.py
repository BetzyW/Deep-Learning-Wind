"""Ejecuta las cadenas del E2 en las dos fuentes con API pública: Semantic Scholar (búsqueda masiva) y arXiv.

Uso (desde literatura/e2): python buscar_apis.py
Salidas: s2_<cadena>.json, arxiv_<cadena>.json y registro_busquedas.csv (fecha, fuente, cadena, consulta, resultados).
"""
import csv, datetime, json, os, time, urllib.error, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

from cadenas import PERIODO, TODAS, arxiv, s2

CAMPOS = "title,year,abstract,venue,externalIds,citationCount,authors,publicationTypes,publicationDate"
NS = {"a": "http://www.w3.org/2005/Atom"}
UA = {"User-Agent": "revision-sistematica-viento/1.0"}


def pedir(url, intentos=6):
    for i in range(intentos):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503):
                raise
            time.sleep(20 * (i + 1))
    raise RuntimeError(f"sin respuesta: {url}")


def buscar_s2(q):
    out, token, total = [], None, None
    while True:
        p = {"query": q, "fields": CAMPOS, "year": f"{PERIODO[0]}-{PERIODO[1]}"}
        if token:
            p["token"] = token
        r = json.loads(pedir("https://api.semanticscholar.org/graph/v1/paper/search/bulk?" + urllib.parse.urlencode(p)))
        total = r.get("total", total)
        out += r.get("data", [])
        token = r.get("token")
        time.sleep(3)
        if not token:
            return out, total


def buscar_arxiv(q):
    out, start = [], 0
    while True:
        u = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"search_query": q, "start": start, "max_results": 200})
        root = ET.fromstring(pedir(u))
        time.sleep(3)
        ents = root.findall("a:entry", NS)
        for e in ents:
            y = int(e.find("a:published", NS).text[:4])
            if PERIODO[0] <= y <= PERIODO[1]:
                out.append(dict(id=e.find("a:id", NS).text, titulo=" ".join(e.find("a:title", NS).text.split()),
                                anio=y, autores=[a.find("a:name", NS).text for a in e.findall("a:author", NS)][:3],
                                resumen=" ".join(e.find("a:summary", NS).text.split())))
        if len(ents) < 200:
            return out, len(out)
        start += 200


if __name__ == "__main__":
    hoy = datetime.date.today().isoformat()
    nuevo = not os.path.exists("registro_busquedas.csv")
    with open("registro_busquedas.csv", "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["fecha", "fuente", "cadena", "consulta", "reportados", "descargados"])
        for n in TODAS:
            for fuente, fn, q in [("Semantic Scholar", buscar_s2, s2(n)), ("arXiv", buscar_arxiv, arxiv(n))]:
                datos, total = fn(q)
                pref = "s2" if fuente == "Semantic Scholar" else "arxiv"
                json.dump(datos, open(f"{pref}_{n}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
                w.writerow([hoy, fuente, n, q, total, len(datos)]); f.flush()
                print(f"{fuente:16s} {n:26s} reportados {total}  descargados {len(datos)}", flush=True)
