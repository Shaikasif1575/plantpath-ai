import httpx, re
h = httpx.get("http://localhost:3000/model", timeout=30).text
m = re.search(r"(\d+) classes reported", h)
print("classes-reported text:", m.group(0) if m else None)
tags = re.findall(r'title="Index \d+">([^<]+)</span>', h)
print("tag count:", len(tags))
print("tags:", tags)
m2 = re.search(r"Class Mapping(.{0,700})", h, re.S)
if m2:
    seg = re.sub(r"\s+", " ", m2.group(1))
    print("section:", seg[:700])
