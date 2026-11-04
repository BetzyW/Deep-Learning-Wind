import urllib.request, urllib.error, urllib.parse, json, time
# Sintaxis de Semantic Scholar: + = AND, | = OR, comillas = frase, * = prefijo
q='"wind speed" + (forecast* | predict*) + ("deep learning" | lstm | "long short-term memory" | gru | "convolutional neural network" | cnn | "multilayer perceptron" | transformer)'
campos="title,year,abstract,venue,externalIds,citationCount,authors,publicationTypes"
out=[]; token=None
while True:
    p={"query":q,"fields":campos,"year":"2018-2026"}
    if token: p["token"]=token
    for intento in range(6):
        try:
            r=json.load(urllib.request.urlopen("https://api.semanticscholar.org/graph/v1/paper/search/bulk?"+urllib.parse.urlencode(p))); break
        except urllib.error.HTTPError as e:
            if e.code!=429: raise
            time.sleep(20*(intento+1))
    out+=r.get("data",[]); token=r.get("token"); time.sleep(2)
    if not token: break
json.dump(out,open("s2.json","w",encoding="utf-8"),ensure_ascii=False,indent=0)
print("total reportado:",r.get("total"),"descargados:",len(out))
