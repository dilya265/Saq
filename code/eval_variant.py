"""Оценка одного варианта модели на Test 1 (замороженные 100 сообщений), результат дописывается в experiments.csv.

  python3 eval_variant.py models/exp_e2 e2          (запускать в GPU-задаче, после обучения)
С 28 октября Test 1 считается DEV-набором: по нему выбираем лучший вариант.
Итоговое честное сравнение нужно делать на новом, ещё не виданном Test 2.
"""
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
if len(sys.argv) < 3:
    sys.exit(__doc__)
model_dir, name = ROOT / sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(ROOT / "data/test_FROZEN.csv", encoding="utf-8-sig", newline="")))

tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model.to(dev).eval()
id2 = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
scam_id = next((i for i, v in id2.items() if v == "scam"), 1)
maxlen = int(os.environ.get("MAXLEN", 128))

pred = []
texts = [r["text"] for r in rows]
for i in range(0, len(texts), 32):
    enc = tok(texts[i:i + 32], truncation=True, max_length=maxlen, padding=True, return_tensors="pt").to(dev)
    with torch.no_grad():
        p = torch.softmax(model(**enc).logits, dim=-1)[:, scam_id]
    pred += (p >= 0.5).long().cpu().tolist()


def count(lang=None):
    sub = [(r, p) for r, p in zip(rows, pred) if lang in (None, r["lang"])]
    sc = [p for r, p in sub if r["label"] == "scam"]
    sf = [p for r, p in sub if r["label"] == "safe"]
    return sum(sc), len(sc), sum(sf), len(sf)


res = {"name": name, "date": datetime.now().strftime("%Y-%m-%d %H:%M")}
for k, lang in (("all", None), ("ru", "ru"), ("kk", "kk")):
    c, n, f, m = count(lang)
    res[f"{k}_caught"], res[f"{k}_false"] = f"{c}/{n}", f"{f}/{m}"
    if k == "all":
        res["accuracy"] = f"{(c + m - f) / (n + m):.0%}"
path = ROOT / "experiments.csv"
new = not path.exists()
with open(path, "a", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(res.keys()))
    if new:
        w.writeheader()
    w.writerow(res)
print(f"{name}: caught {res['all_caught']}, false alarms {res['all_false']}, accuracy {res['accuracy']} | "
      f"ru {res['ru_caught']}/{res['ru_false']} | kk {res['kk_caught']}/{res['kk_false']}")
