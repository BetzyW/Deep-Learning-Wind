"""Búsqueda por citación (snowballing) y revisión de repositorios de código para la novedad del modelo original
(sección 6.3 del enunciado del E3).

* Snowballing en OpenAlex desde los trabajos más cercanos: las tres fuentes del mecanismo (PADRE, TNNR y el ensamble de
  análogos), el paper guía (DLinear) y el trabajo más cercano encontrado (la red pseudo-gemela para potencia eólica).
  Hacia atrás: sus referencias; hacia adelante: los trabajos que los citan (hasta 200 por trabajo). Se marcan como
  candidatos los títulos que combinan el mecanismo (twin, siamese, pairwise, difference, analog, anchor) con el
  problema (wind).
* Repositorios: búsqueda en GitHub (API pública) y en Hugging Face Hub de implementaciones del mecanismo.
Salida: literatura/e3/snowball_{fecha}.csv y literatura/e3/repositorios_{fecha}.csv
Uso: python literatura/e3/snowball_repos.py      (desde la raíz del repositorio)
"""
import csv, json, os, time, urllib.parse, urllib.request

CARPETA = os.path.dirname(os.path.abspath(__file__))
SEMILLAS = {"PADRE (Tynes et al., 2021)": "10.1021/acs.jcim.1c00670",
            "TNNR (Wetzel et al., 2022)": "10.1002/ail2.78",
            "Ensamble de análogos (Delle Monache et al., 2013)": "10.1175/MWR-D-12-00281.1",
            "DLinear, paper guía (Zeng et al., 2023)": "10.1609/aaai.v37i9.26317",
            "Pseudo-gemela para potencia eólica (Yang et al., 2025)": "10.3390/electronics14050887"}
MECANISMO = ["twin", "siamese", "pairwise", "difference", "analog", "anchor"]
REPOS = ["twin neural network regression", "pairwise difference regression", "siamese regression wind",
         "analog ensemble neural network wind", "pairwise difference wind forecasting"]


def obtener(url, intentos=4):
    for k in range(intentos):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "busqueda-academica/1.0",
                                                       "Accept": "application/json"})
            return json.loads(urllib.request.urlopen(req, timeout=60).read())
        except Exception as e:
            err = e
            time.sleep(6 * (k + 1))
    raise err


def obra(doi):
    return obtener(f"https://api.openalex.org/works/https://doi.org/{doi}")


def titulos(ids):
    filas = []
    for i in range(0, len(ids), 50):
        trozo = "|".join(x.rsplit("/", 1)[-1] for x in ids[i:i + 50])
        d = obtener("https://api.openalex.org/works?" + urllib.parse.urlencode(
            {"filter": f"openalex:{trozo}", "per-page": 50, "select": "id,title,publication_year,doi"}))
        filas += d["results"]
    return filas


def citantes(oid, maximo=200):
    d = obtener("https://api.openalex.org/works?" + urllib.parse.urlencode(
        {"filter": f"cites:{oid.rsplit('/', 1)[-1]}", "per-page": maximo,
         "select": "id,title,publication_year,doi", "sort": "cited_by_count:desc"}))
    return d["meta"]["count"], d["results"]


def candidato(t):
    t = (t or "").lower()
    return int(any(k in t for k in MECANISMO) and ("wind" in t or "viento" in t))


if __name__ == "__main__":
    fecha = time.strftime("%Y-%m-%d")
    filas = []
    for nombre, doi in SEMILLAS.items():
        w = obra(doi)
        refs = titulos(w.get("referenced_works", []))
        n_cit, cits = citantes(w["id"])
        for direccion, lista in [("hacia atrás (referencias)", refs), ("hacia adelante (citantes)", cits)]:
            for r in lista:
                filas.append(dict(semilla=nombre, direccion=direccion, anio=r.get("publication_year"),
                                  titulo=r.get("title") or "", id=r.get("doi") or r["id"],
                                  candidato=candidato(r.get("title"))))
        print(f"{nombre}: {len(refs)} referencias, {n_cit} citantes (revisados {len(cits)})", flush=True)
        time.sleep(1)
    with open(os.path.join(CARPETA, f"snowball_{fecha}.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0])); w.writeheader(); w.writerows(filas)
    print("candidatos:", sum(f["candidato"] for f in filas))
    for f in filas:
        if f["candidato"]:
            print("  ", f["semilla"], "|", f["direccion"], "|", f["anio"], f["titulo"][:110])

    repos = []
    for q in REPOS:
        try:
            g = obtener("https://api.github.com/search/repositories?" + urllib.parse.urlencode(
                {"q": q, "per_page": 20, "sort": "stars"}))
            repos += [dict(fuente="GitHub", consulta=q, total=g["total_count"], nombre=r["full_name"],
                           descripcion=(r.get("description") or "")[:200], url=r["html_url"]) for r in g["items"]]
            if not g["items"]:
                repos.append(dict(fuente="GitHub", consulta=q, total=0, nombre="", descripcion="", url=""))
        except Exception as e:
            repos.append(dict(fuente="GitHub", consulta=q, total=-1, nombre="", descripcion=f"falló: {e}", url=""))
        try:
            h = obtener("https://huggingface.co/api/models?" + urllib.parse.urlencode({"search": q, "limit": 20}))
            repos += [dict(fuente="Hugging Face", consulta=q, total=len(h), nombre=m["id"], descripcion="",
                           url=f"https://huggingface.co/{m['id']}") for m in h]
            if not h:
                repos.append(dict(fuente="Hugging Face", consulta=q, total=0, nombre="", descripcion="", url=""))
        except Exception as e:
            repos.append(dict(fuente="Hugging Face", consulta=q, total=-1, nombre="", descripcion=f"falló: {e}",
                              url=""))
        time.sleep(7)                                         # límite de la API pública de GitHub (10 por minuto)
    with open(os.path.join(CARPETA, f"repositorios_{fecha}.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(repos[0])); w.writeheader(); w.writerows(repos)
    for r in repos:
        print(r["fuente"], "|", r["consulta"], "|", r["total"], "|", r["nombre"], "|", r["descripcion"][:90])
