"""Шаг A: проверка и заморозка 100 тестовых сообщений.

  python3 freeze_test.py            проверить data/local.csv и заморозить тест
  python3 freeze_test.py --verify   убедиться, что тестовые строки не изменились
Создаёт data/test_FROZEN.csv (chmod 444) и data/TEST-FROZEN.md (дата, числа, md5).
"""
import csv
import hashlib
import io
import os
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent
D = ROOT / "data"
COLS = ["id", "text", "label", "lang", "type", "source", "split"]


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", str(t).lower())).strip()


def read(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def serialize(rows):
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=COLS, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


local = read(D / "local.csv")
test = [r for r in local if r.get("split") == "test"]

if "--verify" in sys.argv:
    md = (D / "TEST-FROZEN.md").read_text(encoding="utf-8")
    saved = re.search(r"md5: `([0-9a-f]{32})`", md).group(1)
    now = hashlib.md5(serialize(test)).hexdigest()
    print("OK: тестовые строки не изменились" if now == saved else f"ВНИМАНИЕ: тест изменился! было {saved}, стало {now}")
    sys.exit()

errors, warns = [], []
if len(test) != 100:
    errors.append(f"Тестовых строк {len(test)}, нужно 100")
cnt = Counter((r["lang"], r["label"]) for r in test)
for lang in ("ru", "kk"):
    for lab in ("scam", "safe"):
        if cnt[(lang, lab)] != 25:
            errors.append(f"{lang}/{lab}: {cnt[(lang, lab)]}, нужно 25")
ids = [r["id"] for r in test]
if len(set(ids)) != len(ids):
    errors.append("id не уникальны")
for r in test:
    if r["label"] == "scam" and not (r.get("source") or "").strip():
        errors.append(f"{r['id']}: у мошенничества нет source (ссылки)")
    if re.search(r"\+?\d[\d\s\-()]{9,}\d", r["text"]):
        warns.append(f"{r['id']}: похоже на номер телефона или карты, проверьте глазами")

known = set()
for p in (D / "messages.csv", D / "saq_train.csv", D / "messages_public.csv"):
    if p.exists():
        known |= {norm(r["text"]) for r in read(p)}
known |= {norm(r["text"]) for r in local if r.get("split") == "train"}
for r in test:
    if norm(r["text"]) in known:
        errors.append(f"{r['id']}: такой текст уже есть в обучающих данных")

for w in warns:
    print("Предупреждение:", w)
if errors:
    print("НЕ ЗАМОРОЖЕНО. Исправьте:")
    for e in errors:
        print(" -", e)
    sys.exit(1)

fp = D / "test_FROZEN.csv"
if fp.exists():
    os.chmod(fp, 0o644)
data = serialize(test)
fp.write_bytes(data)
os.chmod(fp, 0o444)
md5 = hashlib.md5(data).hexdigest()
(D / "TEST-FROZEN.md").write_text(
    f"# Замороженный тест\n\nДата: {date.today()}\n\nСтрок: {len(test)}\n\n"
    + "\n".join(f"- {l}/{b}: {cnt[(l, b)]}" for l in ("ru", "kk") for b in ("scam", "safe"))
    + f"\n\nmd5: `{md5}`\n\nФайл: data/test_FROZEN.csv (только чтение). Эти строки не менять, не обучать на них, не вставлять в подсказки.\n",
    encoding="utf-8")
print(f"Заморожено {len(test)} строк. md5 {md5}\nТеперь прочитайте все 100 строк вдвоём: верная ли метка, нет ли имён и номеров.")
