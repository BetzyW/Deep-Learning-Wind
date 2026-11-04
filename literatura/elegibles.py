import json, collections
p = json.load(open("tamiz2.json", encoding="utf-8"))
ELEG = [1,7,9,12,13,18,22,23,25,26,29,34,37,41,42,50,51,54,57,71,73,85,88,91,95,100,101,103,107,115,122,126,128,130,134,139,140,149,150,166,178,197,200,208,209]
for i, r in enumerate(p): r["elegible"] = i in ELEG; r["orden"] = i
json.dump(p, open("tamiz2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
e = [p[i] for i in ELEG]
print(len(e), "elegibles")
print(collections.Counter(f for r in e for f in r["familias"]))
