"""Check many messages with the trained detector (models/scam-xlmr). GPU job or CPU (it uses the GPU when there is one).
  sbatch --time 10 --wrap "python3 gpu/predict.py data/test.csv"     -> data/test.pred.csv
  python3 gpu/predict.py "Ваша карта заблокирована, перейдите по ссылке"   (one message, on the CPU, about 1 s)
The CSV needs a text column. If it also has label (safe|scam), it prints: scams caught (recall), false alarms (precision).
"""
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / "models/scam-xlmr"
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForSequenceClassification.from_pretrained(M).to(dev).eval()


def scam_prob(texts, bs=64):
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            x = tok(texts[i:i + bs], truncation=True, max_length=128, padding=True, return_tensors="pt").to(dev)
            out += torch.softmax(model(**x).logits.float(), -1)[:, 1].tolist()
    return out


arg = sys.argv[1]
if not arg.endswith(".csv"):
    p = scam_prob([arg])[0]
    print(f"{'SCAM' if p >= .5 else 'safe'}  (scam probability {p:.2f})")
    sys.exit()
import pandas as pd                                 # only for CSV files
df = pd.read_csv(arg)
df["scam_prob"] = [round(p, 4) for p in scam_prob(df.text.astype(str).tolist())]
df["pred"] = ["scam" if p >= .5 else "safe" for p in df.scam_prob]
out = arg[:-4] + ".pred.csv"; df.to_csv(out, index=False)
print(f"{len(df)} messages -> {out}  ({dev})")
if "label" in df:
    s, n = df[df.label == "scam"], df[df.label == "safe"]
    caught, alarms = (s.pred == "scam").sum(), (n.pred == "scam").sum()
    print(f"caught {caught} of {len(s)} scams, false alarms {alarms} of {len(n)} safe messages")
