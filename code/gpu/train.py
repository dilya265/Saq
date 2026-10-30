"""Train the scam detector: fine-tune xlm-roberta-base on data/messages.csv (a GPU job, about 10-15 min on a V100).
  sbatch --time 30 --wrap "python3 gpu/train.py"          (run from projects/scam-detector)
  EPOCHS=2 ... / LIMIT=2000 (a quick test on 2000 messages)
In:  data/messages.csv (text, label = safe|scam), models/xlm-roberta-base/ (or BASE=FacebookAI/xlm-roberta-base)
Out: models/scam-xlmr/ (model.safetensors + tokenizer), models/scam-xlmr/val_report.txt
10% of the messages (fixed seed 0) are kept for checking (validation); the model never trains on them.
Your real exam stays your own 100 local test messages: check them with gpu/predict.py, once.
"""
import math, os, random, time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent.parent
BASE = os.environ.get("BASE", str(ROOT / "models/xlm-roberta-base"))
OUT = ROOT / os.environ.get("OUT", "models/scam-xlmr")
EPOCHS, LIMIT = float(os.environ.get("EPOCHS", 1)), int(os.environ.get("LIMIT", 0))
MAXLEN, BS, LR, MINUTES = int(os.environ.get("MAXLEN", 128)), 32, float(os.environ.get("LR", 2e-5)), 24          # stop and save before the 30-minute job limit
LABELS = ["safe", "scam"]
dev = "cuda"
t0 = time.time()

df = pd.read_csv(ROOT / "data/messages.csv")
df = df[df.label.isin(LABELS) & df.text.notna()].sample(frac=1, random_state=0)
if LIMIT:
    df = df.head(LIMIT)
n_val = len(df) // 10
val, train = df.head(n_val), df.iloc[n_val:]
print(f"train {len(train)}  check {len(val)}  scam share {train.label.eq('scam').mean():.0%}", flush=True)

tok = AutoTokenizer.from_pretrained(BASE)
model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2, id2label=dict(enumerate(LABELS)),
                                                           label2id={l: i for i, l in enumerate(LABELS)}).to(dev)


def batches(frame, shuffle):
    idx = list(range(len(frame)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), BS):
        part = frame.iloc[idx[i:i + BS]]
        x = tok(list(part.text.astype(str)), truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(dev)
        y = torch.tensor([LABELS.index(l) for l in part.label], device=dev)
        yield x, y


steps = math.ceil(len(train) / BS) * EPOCHS
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
scaler = torch.amp.GradScaler()
random.seed(0); torch.manual_seed(0)
step, done = 0, False
model.train()
while not done:
    for x, y in batches(train, True):
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(**x, labels=y).loss
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
        step += 1
        if step % 200 == 0:
            print(f"step {step}/{int(steps)}  loss {loss.item():.3f}  {(time.time() - t0) / 60:.1f} min", flush=True)
        if step >= steps or time.time() - t0 > MINUTES * 60:
            done = True
            break

model.eval(); tp = fp = fn = tn = 0; per_lang = {}
with torch.no_grad():
    for i, (x, y) in enumerate(batches(val, False)):
        part = val.iloc[i * BS:(i + 1) * BS]
        with torch.autocast("cuda", dtype=torch.float16):
            pred = model(**x).logits.argmax(-1)
        for p, t, lang in zip(pred.tolist(), y.tolist(), part.lang):
            tp += p == t == 1; fp += p == 1 and t == 0; fn += p == 0 and t == 1; tn += p == t == 0
            ok, n = per_lang.get(lang, (0, 0)); per_lang[lang] = (ok + (p == t), n + 1)
rec, prec = tp / max(1, tp + fn), tp / max(1, tp + fp)
report = (f"check set {len(val)} messages: caught {tp} of {tp + fn} scams (recall {rec:.1%}), "
          f"false alarms {fp} of {fp + tn} safe, precision {prec:.1%}\n"
          + "right answers by language: " + ", ".join(f"{k} {a}/{n}" for k, (a, n) in sorted(per_lang.items(), key=lambda kv: -kv[1][1])))
print(report, flush=True)
OUT.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUT); tok.save_pretrained(OUT); (OUT / "val_report.txt").write_text(report + "\n")
print(f"saved {OUT}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
