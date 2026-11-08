"""Verifica el DOI de editorial (Crossref) y el repositorio oficial (API de GitHub) de cada modelo del Top 10.

Uso (desde literatura/e2): python verificar_top10.py → verificacion_top10.json
Crossref se consulta por título; solo se acepta un DOI si el título devuelto coincide (normalizado). Las
conferencias sin DOI de editorial (NeurIPS, ICLR, ICML/PMLR, TMLR) quedan con su DOI de arXiv.
"""
import json, re, time, urllib.parse, urllib.request, urllib.error

UA = {"User-Agent": "revision-sistematica-viento/1.0"}
TOP10 = {
    "Transformer": "google-research/google-research",
    "KAN": "KindXiaoming/pykan",
    "TCN": "locuslab/TCN",
    "STGCN": "VeritasYin/STGCN_IJCAI-18",
    "Chronos": "amazon-science/chronos-forecasting",
    "Mamba": "state-spaces/mamba",
    "GRU": None,
    "Informer": "zhouhaoyi/Informer2020",
    "DLinear": "cure-lab/LTSF-Linear",
    "Moirai": "SalesforceAIResearch/uni2ts",
}


def pedir(url):
    for i in range(5):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(10 * (i + 1))
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            time.sleep(10 * (i + 1))


def norm(t):
    return re.sub(r"[^a-z0-9]", "", str(t).lower())


if __name__ == "__main__":
    orig = json.load(open("originales.json", encoding="utf-8"))
    out = {}
    for m, repo in TOP10.items():
        o = orig[m]
        q = urllib.parse.urlencode({"query.bibliographic": o["titulo"], "rows": 5})
        cr = pedir(f"https://api.crossref.org/works?{q}") or {}
        doi_editorial = None
        for it in (cr.get("message") or {}).get("items", []):
            if norm((it.get("title") or [""])[0]) == norm(o["titulo"]) and not it["DOI"].startswith("10.48550"):
                doi_editorial = it["DOI"]
                break
        info_repo = pedir(f"https://api.github.com/repos/{repo}") if repo else None
        out[m] = dict(titulo=o["titulo"], anio=o["anio"], venue=o["venue"], doi_semantic_scholar=o["doi"],
                      arxiv=o["arxiv"], doi_editorial=doi_editorial, repositorio=repo,
                      repo_existe=bool(info_repo), repo_estrellas=(info_repo or {}).get("stargazers_count"),
                      repo_archivado=(info_repo or {}).get("archived"))
        print(f"{m:12s} DOI editorial: {doi_editorial or '—':32s} repo: {repo or '—'} "
              f"{'OK' if info_repo else ('—' if not repo else 'NO ENCONTRADO')}", flush=True)
        time.sleep(1)
    json.dump(out, open("verificacion_top10.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
