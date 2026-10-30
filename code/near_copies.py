"""Шаг 1 плана среды: проверка похожих текстов (near-copy). Только чтение, ничего не обучает.

  python3 near_copies.py
Для каждой из 100 тестовых строк (data/test_FROZEN.csv) ищет самый похожий обучающий текст в
data/local.csv (split = train), data/saq_train.csv и data/messages.csv.
Сходство: косинус TF-IDF по символьным 3-граммам; цифры, ссылки, [PHONE], [CARD], [CODE] -> один знак #.
Тексты НЕ печатаются. Результат: near_copies.csv, near_copy_check.md и (если файл есть) раздел в results.md.
"""
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent
D = ROOT / "data"
THRESH = 0.8
MAXCH = 500


def norm(t):
    t = str(t).lower()
    t = re.sub(r"https?://\S+|www\.\S+|\[(?:phone|card|code|link|sum|name|shop)\]|\d+", "#", t)
    t = re.sub(r"(#\s*)+", "# ", t)
    return re.sub(r"\s+", " ", t).strip()[:MAXCH]


test = pd.read_csv(D / "test_FROZEN.csv")
local = pd.read_csv(D / "local.csv")
loc_tr = local[local.split == "train"]
saq = pd.read_csv(D / "saq_train.csv")
msg = pd.read_csv(D / "messages.csv")
train_texts = pd.concat([loc_tr.text, saq.text, msg.text]).dropna().astype(str).map(norm)
train_texts = train_texts[train_texts.str.len() > 0].drop_duplicates().tolist()
print(f"Тест: {len(test)} строк, уникальных обучающих текстов: {len(train_texts)}", flush=True)

vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), sublinear_tf=True, dtype=np.float32)
Xtr = vec.fit_transform(train_texts)
Xte = vec.transform(test.text.astype(str).map(norm))
sim = np.zeros(len(test), dtype=np.float32)
for i in range(0, Xte.shape[0], 20):
    block = (Xte[i:i + 20] @ Xtr.T).toarray()
    sim[i:i + 20] = block.max(axis=1)

out = test[["id", "lang", "label", "source"]].copy()
out["similarity"] = sim.round(3)
out.to_csv(ROOT / "near_copies.csv", index=False)
near = out[out.similarity >= THRESH]
n_near = len(near)
print(f"near-copies (>= {THRESH}): {n_near} of {len(test)}")

lines = [f"\n## Near-copy check ({date.today()})", "",
         f"Near-copies (character 3-gram TF-IDF cosine >= {THRESH}) between a test message and any training text: **{n_near} of {len(test)}**.", "",
         "Test rows with similarity >= 0.8, per language and label:", "",
         "| Language | Label | Near-copies | Max similarity |", "|---|---|---|---|"]
for (l, b), g in out.groupby(["lang", "label"]):
    lines.append(f"| {l} | {b} | {int((g.similarity >= THRESH).sum())} of {len(g)} | {g.similarity.max():.2f} |")

tr = pd.concat([loc_tr[["lang", "label"]], saq[["lang", "label"]]])
lines += ["", "Training rows, local + Saq (each counted once; they are repeated 10x in data/messages.csv):", "",
          "| Language | scam | safe |", "|---|---|---|"]
for l in ("ru", "kk"):
    g = tr[tr.lang == l]
    lines.append(f"| {l} | {int((g.label == 'scam').sum())} | {int((g.label == 'safe').sum())} |")

lines += ["", f"Scores WITHOUT the {n_near} near-copy test rows (caught of scams / false alarms of safe):", "",
          "| Method | Language | Caught | False alarms |", "|---|---|---|---|"]
keep = out[out.similarity < THRESH][["id", "lang", "label"]]
for name, f in (("TF-IDF + LogReg", "baseline"), ("Saq rules", "saq_rules"), ("XLM-R fine-tuned", "scam_xlmr")):
    p = ROOT / f"preds_{f}.csv"
    if not p.exists():
        continue
    d = keep.merge(pd.read_csv(p), on="id")
    for l in ("ru", "kk"):
        g = d[d.lang == l]
        sc, sf = g[g.label == "scam"], g[g.label == "safe"]
        lines.append(f"| {name} | {l} | {int((sc.pred == 'scam').sum())} / {len(sc)} | {int((sf.pred == 'scam').sum())} / {len(sf)} |")
lines.append("")
text = "\n".join(lines)
(ROOT / "near_copy_check.md").write_text(text, encoding="utf-8")
print("Записано: near_copy_check.md")
rp = ROOT / "results.md"
if rp.exists():
    with open(rp, "a", encoding="utf-8") as fh:
        fh.write(text)
    print("Добавлено в results.md: раздел Near-copy check")
else:
    print("results.md ещё нет (его делает Assem): раздел лежит в near_copy_check.md, добавьте его в results.md позже.")
