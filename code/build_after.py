"""Шаг 4 (часть 1): добавляет локальные обучающие сообщения в data/messages.csv (каждое 10 раз).

  python3 build_after.py
Делает копии: data/messages_public.csv и models/scam-xlmr-before (если их ещё нет).
messages.csv каждый раз собирается заново из messages_public.csv, поэтому запускать можно повторно.
Строки split = test не добавляются никогда.
"""
import csv
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
D = ROOT / "data"
COLS = ["id", "text", "lang", "label", "type", "source", "split"]
TIMES = 10


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", str(t).lower())).strip()


def read(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


pub = D / "messages_public.csv"
if not pub.exists():
    shutil.copy(D / "messages.csv", pub)
    print("Копия: data/messages_public.csv")
before = ROOT / "models/scam-xlmr-before"
if not before.exists() and (ROOT / "models/scam-xlmr").exists():
    shutil.copytree(ROOT / "models/scam-xlmr", before)
    print("Копия модели: models/scam-xlmr-before")

local = read(D / "local.csv")
test_texts = {norm(r["text"]) for r in local if r.get("split") == "test"}
new = []
for i, r in enumerate(r for r in local if r.get("split") == "train"):
    new.append({"id": f"loc_{i:04d}", "text": r["text"], "lang": r.get("lang", "ru"), "label": r["label"],
                "type": r.get("type", "none"), "source": r.get("source", "local"), "split": "train"})
n_local = len(new)
for i, r in enumerate(read(D / "saq_train.csv")):
    new.append({"id": f"saq_{i:04d}", "text": r["text"], "lang": r["lang"], "label": r["label"],
                "type": r.get("type", "none"), "source": "synthetic", "split": "train"})

bad = [r["id"] for r in new if norm(r["text"]) in test_texts]
if bad:
    sys.exit(f"СТОП: эти обучающие строки совпадают с тестом: {bad[:5]}. Удалите их из обучения.")

with open(pub, encoding="utf-8-sig", newline="") as f:
    base = list(csv.DictReader(f))
with open(D / "messages.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
    w.writeheader()
    w.writerows(base)
    for k in range(TIMES):
        for r in new:
            w.writerow({**r, "id": f"{r['id']}_x{k}"})
print(f"Публичных строк: {len(base)}. Добавлено {len(new)} локальных/Saq строк x{TIMES} (локальных: {n_local}).")
print("Дальше: sbatch --time 30 --wrap \"python3 gpu/train.py\"")
