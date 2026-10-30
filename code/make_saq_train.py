"""Шаг 1 (часть 1): собирает data/saq_train.csv из моих двух файлов.

  python3 make_saq_train.py
Читает ~/work/saq_synthetic.csv и ~/work/saq_kazakh_1.csv (или saq_kazakh.csv),
пишет data/saq_train.csv: text, label (ok -> safe), lang, type (= scheme), source = synthetic.
"""
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOME = Path.home() / "work"
files = [HOME / "saq_synthetic.csv", HOME / "saq_kazakh_1.csv"]
if not files[1].exists():
    files[1] = HOME / "saq_kazakh.csv"

rows, seen = [], set()
for f in files:
    if not f.exists():
        print("Нет файла:", f)
        continue
    with open(f, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            text = (r.get("text") or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            label = "scam" if r["label"].strip() == "scam" else "safe"
            rows.append({"text": text, "label": label, "lang": r.get("lang", "ru"),
                         "type": (r.get("scheme") or "").strip() or "none", "source": "synthetic"})

os.makedirs(ROOT / "data", exist_ok=True)
with open(ROOT / "data/saq_train.csv", "w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["text", "label", "lang", "type", "source"])
    w.writeheader()
    w.writerows(rows)
print(f"Записано {len(rows)} строк в data/saq_train.csv")
