"""Сравнение 4 методов: baseline (TF-IDF + LogReg), saq_rules, xlm-roberta-base БЕЗ дообучения, scam-xlmr.

Запускать из projects/scam-detector.
  python3 eval_compare.py --mode val --no-xlmr          быстрая проверка на отложенных 10% (почти все английские)
  python3 eval_compare.py --mode test --tag before      экзамен на 100 настоящих сообщениях (до дообучения)
  python3 eval_compare.py --mode test --tag after       то же после дообучения с локальными данными
GPU:  sbatch --time=15 --wrap "python3 eval_compare.py --mode test --tag before"
В режиме test скрипт САМ останавливается, если в обучении есть тестовый текст
или тест не совпадает с data/test_FROZEN.csv.
Результаты: results_<tag>.md и results_<tag>.json, а также preds_<метод>.csv (id,pred).
"""
import argparse
import json
import random
import urllib.request
import os
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parent
for p in (ROOT, ROOT.parent.parent, Path.home() / "work"):   # saq_check.py: рядом или в ~/work
    sys.path.insert(0, str(p))
LABELS = ["safe", "scam"]                      # 0 = safe, 1 = scam (как в gpu/train.py)
MODEL_DIR = ROOT / "models/scam-xlmr"
BASE = os.environ.get("BASE", str(ROOT / "models/xlm-roberta-base"))
MAXLEN, BS = 128, 32

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["val", "test"], default="val")
ap.add_argument("--tag", default="run")
ap.add_argument("--no-xlmr", action="store_true")
ap.add_argument("--chat", action="store_true", help="5-й метод: класс-модель (только в терминале, не в GPU-задаче)")
args = ap.parse_args()


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", str(t).lower())).strip()


df = pd.read_csv(ROOT / "data/messages.csv")
df = df[df.label.isin(LABELS) & df.text.notna()].sample(frac=1, random_state=0)
n_val = len(df) // 10
val, train = df.head(n_val), df.iloc[n_val:]

if args.mode == "val":
    test = val
else:
    lp, fp = ROOT / "data/local.csv", ROOT / "data/test_FROZEN.csv"
    if not lp.exists():
        sys.exit("СТОП: нет data/local.csv (100 настоящих тестовых сообщений ещё не собрано).")
    if not fp.exists():
        sys.exit("СТОП: нет data/test_FROZEN.csv. Сначала: python3 freeze_test.py")
    local = pd.read_csv(lp)
    test = local[(local.split == "test") & local.label.isin(LABELS) & local.text.notna()]
    frozen = pd.read_csv(fp)
    key = lambda d: sorted(zip(d["id"].astype(str), d["text"].astype(str), d["label"].astype(str)))
    if key(test) != key(frozen):
        sys.exit("СТОП: тестовые строки в data/local.csv не совпадают с data/test_FROZEN.csv.")
    test_norm = {norm(t) for t in test.text}
    sources = {"data/messages.csv": df.text, "split=train из data/local.csv": local[local.split == "train"].text}
    sp = ROOT / "data/saq_train.csv"
    if sp.exists():
        sources["data/saq_train.csv"] = pd.read_csv(sp).text
    for name, col in sources.items():
        leak = [t for t in col.astype(str) if norm(t) in test_norm]
        if leak:
            sys.exit(f"СТОП: тестовый текст есть в обучающих данных ({name}): {leak[0][:80]!r}")
    print("Проверка утечки пройдена: тест не пересекается с обучением и совпадает с test_FROZEN.csv.")

print(f"Режим: {args.mode} | обучение baseline: {len(train)} | проверка: {len(test)}", flush=True)
y_true = test.label.map(LABELS.index).tolist()
results, preds_all = {}, {}


def stats(frame, preds):
    t = frame.assign(y=y_true, p=preds)
    out = {}
    groups = [("all", t)] + [(l, g) for l, g in t.groupby("lang")] if "lang" in t.columns else [("all", t)]
    for k, g in groups:
        sc, sf = g[g.y == 1], g[g.y == 0]
        out[k] = {"caught": int((sc.p == 1).sum()), "scams": len(sc), "false": int((sf.p == 1).sum()), "safe": len(sf)}
    return out


def report(key, title, preds):
    print("=" * 60)
    print("РЕЗУЛЬТАТ:", title)
    print(classification_report(y_true, preds, labels=[0, 1], target_names=["Safe (0)", "Scam (1)"], digits=4, zero_division=0))
    results[key] = stats(test, preds)
    preds_all[key] = preds
    if args.mode == "test":
        pd.DataFrame({"id": test["id"], "pred": ["scam" if p else "safe" for p in preds]}).to_csv(ROOT / f"preds_{key}.csv", index=False)


# 1. baseline
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=300000, sublinear_tf=True)
base = LogisticRegression(max_iter=1000, class_weight="balanced").fit(vec.fit_transform(train.text.astype(str)), train.label.map(LABELS.index))
report("baseline", "BASELINE (TF-IDF + Logistic Regression)", base.predict(vec.transform(test.text.astype(str))).tolist())

# 2. правила Saq: красный = scam, жёлтый и зелёный = safe
try:
    from saq_check import check
    report("saq_rules", "SAQ RULES (красный = scam)", [1 if check(t)[0] == "red" else 0 for t in test.text.astype(str)])
except ImportError:
    print("Предупреждение: saq_check.py не найден (положите его рядом или в ~/work). Метод saq_rules пропущен.")



# 5. класс-модель (chat_model): один запрос на сообщение, ответ только JSON. Ключ берётся из $FRP_KEY, не печатается.
extra = {}
if args.chat:
    key = os.environ.get("FRP_KEY")
    if not key:
        sys.exit("СТОП: переменная FRP_KEY не задана.")
    url = os.environ.get("CHAT_URL", "http://127.0.0.1:19040/v1/chat/completions")
    model_name = os.environ.get("CHAT_MODEL", "dsv41")
    saq = pd.read_csv(ROOT / "data/saq_train.csv")          # примеры ТОЛЬКО из обучающих данных
    rng = random.Random(0)
    shots = []
    for lab in ("scam", "safe"):
        for lang, n in (("kk", 2), ("ru", 1)):
            pool = saq[(saq.label == lab) & (saq.lang == lang)].text.tolist()
            shots += [(t, lab) for t in rng.sample(pool, min(n, len(pool)))]
    rng.shuffle(shots)
    system = (
        "You are a scam detector for text messages in Kazakhstan (Russian, Kazakh or mixed). "
        "Decide if the message is a scam (fraud, phishing, fake bank or payout, fake prize, fake job, fake investment, "
        "a relative asking for money) or safe. Real service messages (purchase notices, login codes with a warning not to share them, "
        "delivery notices) and normal chat messages are safe. "
        "Answer with JSON only and nothing else: {\"label\": \"scam\" or \"safe\", \"red_flags\": [short phrases copied from the message]}. "
        "For safe messages red_flags is [].\n\nExamples (message -> label):\n"
        + "\n".join(f"- {t} -> {l}" for t, l in shots)
    )
    (ROOT / "chat_prompt.txt").write_text(system, encoding="utf-8")   # сам запрос без ключа; для статьи
    import re as _re

    def ask(text):
        body = json.dumps({"model": model_name, "temperature": 0, "max_tokens": 300,
                           "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}]}).encode()
        req = urllib.request.Request(url, body, {"Content-Type": "application/json", "Authorization": "Bearer " + key})
        for _ in range(3):
            try:
                with urllib.request.urlopen(req, timeout=90) as r:
                    return json.loads(r.read())["choices"][0]["message"]["content"] or ""
            except Exception:
                continue
        return ""

    preds_chat, bad = [], 0
    for i, t in enumerate(test.text.astype(str), 1):
        m = _re.search(r"\{.*\}", ask(t), _re.S)
        lab = None
        if m:
            try:
                lab = str(json.loads(m.group(0)).get("label", "")).strip().lower()
            except Exception:
                lab = None
        if lab not in ("scam", "safe"):
            bad += 1
            lab = "safe"                 # плохой ответ считается safe (по плану)
        preds_chat.append(1 if lab == "scam" else 0)
        if i % 20 == 0:
            print(f"chat_model: {i}/{len(test)}", flush=True)
    report("chat_model", f"CHAT MODEL ({model_name})", preds_chat)
    extra["chat_bad_answers"] = bad
    print(f"chat_model: плохих ответов (не JSON): {bad} из {len(test)}")

# 3 и 4. трансформеры
def run_transformer(key, title, path, untuned=False):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    if untuned:
        torch.manual_seed(0)  # слой классификации новый и случайный
        model = AutoModelForSequenceClassification.from_pretrained(
            path, num_labels=2, id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)})
    else:
        model = AutoModelForSequenceClassification.from_pretrained(path)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()
    id2 = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    scam_id = next((i for i, v in id2.items() if v == "scam"), 1)
    texts, preds = test.text.astype(str).tolist(), []
    for i in range(0, len(texts), BS):
        enc = tok(texts[i:i + BS], truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(dev)
        with torch.no_grad():
            pr = torch.softmax(model(**enc).logits, dim=-1)[:, scam_id]
        preds += (pr >= 0.5).long().cpu().tolist()
    report(key, title, preds)


if not args.no_xlmr:
    bp = BASE if os.path.isdir(BASE) else "FacebookAI/xlm-roberta-base"
    run_transformer("roberta_untuned", "xlm-roberta-base БЕЗ дообучения (точка отсчёта)", bp, untuned=True)
    run_transformer("scam_xlmr", "scam-xlmr (наш дообученный XLM-RoBERTa)", MODEL_DIR)

# сводка
lines = ["СВОДКА (поймано мошенничеств / ложные тревоги; всегда смотрите на оба числа)"]
for key, r in results.items():
    for k, v in r.items():
        label = key if k == "all" else f"  {key} {k}"
        lines.append(f"{label:<22} поймано {v['caught']}/{v['scams']} | ложных {v['false']}/{v['safe']}")
print("=" * 60)
print("\n".join(lines))
if args.mode == "test":
    (ROOT / f"results_{args.tag}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    note = "".join(f"\n{k}: {v}" for k, v in extra.items())
    (ROOT / f"results_{args.tag}.md").write_text("# Результаты (" + args.tag + ")\n\n```\n" + "\n".join(lines) + note + "\n```\n", encoding="utf-8")
    print(f"Сохранено: results_{args.tag}.md и results_{args.tag}.json")
