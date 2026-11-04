import urllib.request, urllib.parse, json, time, xml.etree.ElementTree as ET
ns={"a":"http://www.w3.org/2005/Atom"}
q='ti:"wind speed" AND (ti:forecast* OR ti:predict*) AND (abs:"deep learning" OR abs:LSTM OR abs:GRU OR abs:"convolutional" OR abs:transformer OR abs:"neural network")'
out=[]; start=0
while True:
    u="http://export.arxiv.org/api/query?"+urllib.parse.urlencode({"search_query":q,"start":start,"max_results":200})
    root=ET.fromstring(urllib.request.urlopen(u).read()); time.sleep(3)
    ents=root.findall("a:entry",ns)
    for e in ents:
        y=int(e.find("a:published",ns).text[:4])
        if y<2018: continue
        out.append(dict(id=e.find("a:id",ns).text,titulo=" ".join(e.find("a:title",ns).text.split()),anio=y,
            autores=[a.find("a:name",ns).text for a in e.findall("a:author",ns)][:3],resumen=" ".join(e.find("a:summary",ns).text.split())))
    if len(ents)<200: break
    start+=200
json.dump(out,open("arxiv.json","w",encoding="utf-8"),ensure_ascii=False,indent=0)
print(len(out))
