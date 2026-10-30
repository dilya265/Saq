"""Проверка дообученной модели xlm-roberta-base.

Один текст:
  python3 predict_xlmr.py MODEL_DIR "Служба безопасности Kaspi: назовите код из SMS"

Весь тестовый набор (строки split = test из data/local.csv или data/messages.csv):
  python3 predict_xlmr.py MODEL_DIR
  -> создаёт preds_xlmr.csv (id,pred), потом:  python3 eval.py preds_xlmr.csv

MODEL_DIR: папка, куда обучение сохранило модель (в ней config.json и model.safetensors).
Если задача большая, запускайте через GPU:  sbatch --wrap "python3 predict_xlmr.py MODEL_DIR"
"""
import csv
import os
import sys

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

if len(sys.argv) < 2:
    sys.exit(__doc__)

model_dir = sys.argv[1]
tok = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

# какой номер класса означает scam (по умолчанию 1)
id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
scam_id = next((i for i, v in id2label.items() if v == "scam"), 1)


def scam_prob(texts, batch=16):
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], truncation=True, max_length=256, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1)[:, scam_id]
        out += probs.cpu().tolist()
    return out


if len(sys.argv) > 2:
    text = " ".join(sys.argv[2:])
    p = scam_prob([text])[0]
    print(f"scam: {p:.0%}  ->", "🔴 похоже на мошенничество" if p >= 0.5 else "🟢 признаков нет")
    sys.exit()

path = next((p for p in ("data/local.csv", "data/messages.csv") if os.path.exists(p)), None)
rows = []
if path:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("split") == "test"]
if not rows:
    sys.exit("Нет строк split = test. Тестовый набор (data/local.csv) ещё не собран.")

probs = scam_prob([r["text"] for r in rows])
with open("preds_xlmr.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "pred", "prob"])
    for r, p in zip(rows, probs):
        w.writerow([r["id"], "scam" if p >= 0.5 else "safe", f"{p:.3f}"])
print(f"Готово: {len(rows)} предсказаний в preds_xlmr.csv. Дальше: python3 eval.py preds_xlmr.csv")
