"""Mueve el trabajo del E2 entre el PC y Google Colab sin perder ni duplicar resultados.

  python src/sincronizar.py empaquetar DESTINO.zip
      Crea un .zip con los datos (datos/) y los resultados del E2 (resultados/e2/) para subirlo a Google Drive.
  python src/sincronizar.py unir CARPETA_DE_COLAB
      Une en resultados/e2/busqueda/ lo que llegó de Colab (la carpeta resultados_e2/busqueda descargada de Drive).
      Las filas se unen por su identificador (id en el ajuste y la fase 1, clave en la fase 2): las que ya están en el
      PC se conservan y solo se agregan las nuevas, de modo que se puede unir varias veces sin duplicar.
"""
import csv, json, os, shutil, sys, zipfile

CARPETA = "resultados/e2/busqueda"
CLAVES = {"_hp.csv": "id", "_fase1.csv": "id", "_fase2.csv": "clave"}


def empaquetar(destino):
    n = 0
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:
        for base in ["datos", "resultados/e2"]:
            for raiz, _, archivos in os.walk(base):
                for a in archivos:
                    if a.endswith((".log", ".err")):
                        continue
                    ruta = os.path.join(raiz, a)
                    z.write(ruta, ruta.replace("\\", "/")); n += 1
    print(f"{n} archivos → {destino} ({os.path.getsize(destino) / 1e6:.0f} MB)")


def leer(ruta):
    with open(ruta, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def unir(origen):
    os.makedirs(CARPETA, exist_ok=True)
    for a in sorted(os.listdir(origen)):
        src, dst = os.path.join(origen, a), os.path.join(CARPETA, a)
        sufijo = next((s for s in CLAVES if a.endswith(s)), None)
        if sufijo:
            clave = CLAVES[sufijo]
            locales = leer(dst) if os.path.exists(dst) else []
            vistos = {r[clave] for r in locales}
            nuevas = [r for r in leer(src) if r[clave] not in vistos]
            if not nuevas:
                continue
            campos = list((locales or nuevas)[0])
            with open(dst, "a" if locales else "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=campos)
                if not locales:
                    w.writeheader()
                w.writerows(nuevas)
            print(f"{a}: +{len(nuevas)} filas (total {len(locales) + len(nuevas)})")
        elif a == "hardware.jsonl":
            ya = set(open(dst, encoding="utf-8").read().splitlines()) if os.path.exists(dst) else set()
            nuevas = [l for l in open(src, encoding="utf-8").read().splitlines() if l and l not in ya]
            with open(dst, "a", encoding="utf-8") as f:
                f.writelines(l + "\n" for l in nuevas)
            print(f"{a}: +{len(nuevas)} registros")
        elif a.endswith(".json"):
            if os.path.exists(dst) and json.load(open(dst)) != json.load(open(src)):
                print(f"OJO {a}: distinto en el PC y en Colab; se conserva el del PC")
            elif not os.path.exists(dst):
                shutil.copy2(src, dst); print(f"{a}: copiado")
        elif a.endswith((".log", ".err")):
            shutil.copy2(src, os.path.join(CARPETA, "colab_" + a)); print(f"{a}: copiado como colab_{a}")


if __name__ == "__main__":
    {"empaquetar": empaquetar, "unir": unir}[sys.argv[1]](sys.argv[2])
