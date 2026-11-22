"""Descarga los conjuntos adicionales del E3 desde sus fuentes oficiales y verifica cada archivo con SHA-256.

Los datos no se versionan (ETTh1 es CC BY-ND y no se redistribuyen versiones modificadas); este script los obtiene
igual que el 21-nov-2026 y compara cada archivo con `experiments/e3_sha256.csv`. Si una huella no coincide, avisa (la
fuente cambió) y no se detiene: la diferencia se reporta.
* ETTh1: GitHub zhouhaoyi/ETDataset, ETT-small/ETTh1.csv.
* SDWPF: figshare doi:10.6084/m9.figshare.24798654 (archivos 46005813 y 46005807; 334 MB en total).
* NOAA ISD-Lite, Kansas: AWS Open Data, s3://noaa-isd-pds/isd-lite/data/{año}/{USAF}-{WBAN}-{año}.gz. La lista de las
  34 estaciones candidatas (regla de selección en experiments/conjuntos_adicionales.md) está en
  experiments/e3_estaciones_kansas.csv; por defecto se bajan solo las 10 que se usan (semilla 11, la misma de
  `e3_conjuntos.py`) y los años 2021–2024. Con --todas se bajan las 34 estaciones de 2019 a 2024 (204 archivos).
Uso: python src/e3_descargar.py [--todas]      y luego  python src/e3_conjuntos.py
"""
import csv, hashlib, os, shutil, sys, urllib.request
import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(RAIZ, "datos", "e3")
N_NOAA = 10                                     # igual que en e3_conjuntos.py
FUENTES = {
    "etth1/ETTh1.csv": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "sdwpf/sdwpf_245days_v1.csv": "https://ndownloader.figshare.com/files/46005813",
    "sdwpf/sdwpf_baidukddcup2022_turb_location.csv": "https://ndownloader.figshare.com/files/46005807",
}


def sha(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def bajar(rel, url, esperadas):
    ruta = os.path.join(D, rel)
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    if not os.path.exists(ruta):
        req = urllib.request.Request(url, headers={"User-Agent": "python-urllib (investigacion academica)"})
        with urllib.request.urlopen(req, timeout=300) as r, open(ruta + ".parcial", "wb") as f:
            shutil.copyfileobj(r, f)
        os.replace(ruta + ".parcial", ruta)
    h = sha(ruta)
    ok = esperadas.get(rel) == h
    print(f"{'ok ' if ok else 'DIFERENTE'} {rel}" + ("" if ok else f" (esperada {esperadas.get(rel, '—')[:12]}…, "
                                                                  f"obtenida {h[:12]}…)"), flush=True)
    return ok


if __name__ == "__main__":
    todas = "--todas" in sys.argv
    with open(os.path.join(RAIZ, "experiments", "e3_sha256.csv"), newline="") as f:
        esperadas = {r["archivo"]: r["sha256"] for r in csv.DictReader(f)}
    est = list(csv.DictReader(open(os.path.join(RAIZ, "experiments", "e3_estaciones_kansas.csv"), encoding="utf-8")))
    os.makedirs(os.path.join(D, "noaa_kansas"), exist_ok=True)
    shutil.copy(os.path.join(RAIZ, "experiments", "e3_estaciones_kansas.csv"),
                os.path.join(D, "noaa_kansas", "estaciones_candidatas.csv"))
    if not todas:
        est = [est[i] for i in np.sort(np.random.default_rng(11).choice(len(est), N_NOAA, replace=False))]
    fuentes = dict(FUENTES)
    for r in est:
        for a in (range(2019, 2025) if todas else range(2021, 2025)):
            nombre = f"{r['USAF']}-{r['WBAN']}-{a}.gz"
            fuentes[f"noaa_kansas/isd_lite/{nombre}"] = f"https://noaa-isd-pds.s3.amazonaws.com/isd-lite/data/{a}/{nombre}"
    malos = [rel for rel, url in fuentes.items() if not bajar(rel, url, esperadas)]
    print(f"{len(fuentes) - len(malos)} de {len(fuentes)} archivos con la huella registrada"
          + (f"; distintos: {malos}" if malos else ""), flush=True)
