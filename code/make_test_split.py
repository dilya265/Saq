"""Выбирает 100 тестовых строк в data/local.csv: по 25 на каждую пару (язык, метка).

  python3 make_test_split.py
Выбор случайный (seed = 0), без чтения текстов, чтобы тест не подгонялся.
Остальные строки остаются split = train. Копия до выбора: data/local_before_split.csv.
Запускать один раз, пока в файле нет строк split = test.
"""
import csv
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

D = Path(__file__).resolve().parent / "data"
src = D / "local.csv"
with open(src, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
fields = list(rows[0].keys())

if any(r["split"] == "test" for r in rows):
    sys.exit("СТОП: в local.csv уже есть строки split = test. Ничего не меняю.")

cells = {}
for i, r in enumerate(rows):
    cells.setdefault((r["lang"], r["label"]), []).append(i)

need = [(l, b) for l in ("ru", "kk") for b in ("scam", "safe")]
problems = [f"{l}/{b}: {len(cells.get((l, b), []))}, нужно минимум 25" for l, b in need if len(cells.get((l, b), [])) < 25]
if problems:
    sys.exit("СТОП: не хватает сообщений:\n - " + "\n - ".join(problems))

shutil.copy(src, D / "local_before_split.csv")
rng = random.Random(0)
for key in need:
    for i in rng.sample(cells[key], 25):
        rows[i]["split"] = "test"

with open(src, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

print("Готово. Строки по частям:")
for (s, l, b), n in sorted(Counter((r["split"], r["lang"], r["label"]) for r in rows).items()):
    print(f"  {s:5} {l} {b:5} {n}")
print("Дальше: python3 freeze_test.py")
