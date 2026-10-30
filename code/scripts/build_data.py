"""Build data/messages.csv from the three public Hugging Face datasets.

Run: python3 scripts/build_data.py
Needs: data/raw/ (hf download commands are in the report / README).
"""

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "messages.csv"
META = ROOT / "data" / "messages.meta.json"
MAX_CHARS = 2000

SOURCES = {
    "sms_spam": "ucirvine/sms_spam",
    "difraud_sms": "redasers/difraud (sms)",
    "difraud_phishing": "redasers/difraud (phishing)",
    "difraud_job_scams": "redasers/difraud (job_scams)",
    "fz_junkmail": "FredZhang7/all-scam-spam",
}

SCAM_TYPES = {
    "sms_spam": "other",
    "difraud_sms": "other",
    "difraud_phishing": "phishing_link",
    "difraud_job_scams": "job_offer",
    "fz_junkmail": "other",
}

KK_LETTERS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
CYRILLIC = re.compile(r"[А-Яа-яЁё]")
LATIN = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")
OTHER_SCRIPTS = re.compile(r"[\u0600-\u06FF\u0400-\u04FF\u4E00-\u9FFF\u3040-\u30FF\uAC00-\uD7AF\u0E00-\u0E7F\u0590-\u05FF\u0370-\u03FF]")


def language(text):
    """Rough language guess: Cyrillic -> ru/kk, Latin -> stopword vote, else other."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "other"
    cyr = sum(1 for c in letters if CYRILLIC.match(c) or c in KK_LETTERS)
    lat = sum(1 for c in letters if LATIN.match(c))
    if cyr > lat * 0.2:
        if any(c in KK_LETTERS for c in text):
            return "kk"
        return "ru"
    if lat == 0:
        return "other"
    return latin_language(text)


HINTS = {
    "en": "the and that this with have from they what when your are was were will would there their about just know like can not but you for get got now our out all she his her has had been more one who how why than then them these those should could time please".split(),
    "es": "pero como para este esta cuando muy porque todo bien hecho dia gracias hola tambien usted nosotros donde siempre nunca ahora aqui algo bueno mejor quiero puedes tienes hacer".split(),
    "it": "che non sono ciao grazie anche questo questa della degli piu perche tutto bene fatto giorno adesso ancora sempre molto poco voglio puoi hai siamo nella sul alla".split(),
    "fr": "les des est une dans pour avec mais vous nous cette aussi tres merci bonjour etre avoir fait tout plus sans sous entre quand comme leur notre votre".split(),
    "de": "und ist nicht ein eine mit sich auf fur auch aber wird sind sehr danke hallo haben werden kann muss oder wenn dann noch nur schon bei aus nach".split(),
    "pt": "nao uma para com isso muito obrigado voce tambem esta porque mas foi ser tem pelo pela dos das seu sua aqui agora ainda bem todo".split(),
    "nl": "het een van ik dat die niet zijn met voor maar aan ook nog wel geen deze wordt hebben kan moet bij".split(),
    "tr": "icin bir bu ve cok daha gibi ama degil var yok merhaba tesekkur lutfen ben sen biz siz nasil neden sey ile".split(),
    "id": "yang dan ini itu dengan untuk tidak dari dalam akan pada saya kami ada juga sudah bisa karena kalau tetapi atau sangat".split(),
    "tl": "ang mga hindi para ako siya ito pero dahil kung lang naman yung may wala gusto kami kayo salamat".split(),
}
HINT_SETS = {lang: set(words) for lang, words in HINTS.items()}
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def fold(word):
    return "".join(
        c for c in unicodedata.normalize("NFKD", word.lower()) if not unicodedata.combining(c)
    ).replace("ı", "i")


def latin_language(text):
    words = [fold(w) for w in WORD.findall(text)]
    if not words:
        return "other"
    scores = {lang: len(set(words) & hints) for lang, hints in HINT_SETS.items()}
    english = scores["en"]
    best = max(scores, key=scores.get)
    if best == "en" or scores[best] < 2 or scores[best] < english + 2:
        return "en"
    if scores[best] == max(score for lang, score in scores.items() if lang != best):
        return "en"
    return best


def clean_text(value):
    text = "" if value is None else str(value)
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def add_frame(rows, source, frame, text_col, label_col, label_map):
    for text, raw_label in zip(frame[text_col], frame[label_col]):
        text = clean_text(text)
        if len(text) < 2:
            continue
        label = label_map.get(str(raw_label).strip().lower())
        if label is None:
            raise ValueError(f"{source}: unknown label {raw_label!r}")
        rows.append(
            {
                "text": text,
                "lang": language(text),
                "label": label,
                "type": SCAM_TYPES[source] if label == "scam" else "none",
                "source": source,
            }
        )


def truncate(text, limit=MAX_CHARS):
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    if space > limit * 0.8:
        cut = cut[:space]
    return cut.rstrip()


def main():
    rows = []
    raw_counts = {}

    sms = pd.read_parquet(RAW / "sms_spam/plain_text/train-00000-of-00001.parquet")
    add_frame(rows, "sms_spam", sms, "sms", "label", {"1": "scam", "0": "safe"})
    raw_counts["sms_spam"] = len(rows)
    print(f"ucirvine/sms_spam: {len(rows)} rows")

    for subset in ["sms", "phishing", "job_scams"]:
        source = f"difraud_{subset}"
        start = len(rows)
        for split in ["train", "validation", "test"]:
            path = RAW / "difraud" / subset / f"{split}.jsonl"
            records = []
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        records.append(json.loads(line))
            frame = pd.DataFrame(records)
            add_frame(rows, source, frame, "text", "label", {"1": "scam", "0": "safe"})
        raw_counts[source] = len(rows) - start
        print(f"redasers/difraud {subset}: {len(rows) - start} rows")

    start = len(rows)
    fz = pd.read_csv(RAW / "fz/junkmail_dataset.csv")
    add_frame(rows, "fz_junkmail", fz, "text", "is_spam", {"1": "scam", "0": "safe"})
    raw_counts["fz_junkmail"] = len(rows) - start
    print(f"FredZhang7/all-scam-spam: {raw_counts['fz_junkmail']} rows")

    frame = pd.DataFrame(rows)
    length_before = {
        label: {
            "mean": round(float(group.str.len().mean()), 1),
            "median": int(group.str.len().median()),
            "p90": int(group.str.len().quantile(0.90)),
            "p99": int(group.str.len().quantile(0.99)),
            "max": int(group.str.len().max()),
        }
        for label, group in frame.groupby("label")["text"]
    }
    word_counts = {
        label: {
            "mean": round(float(group.str.split().str.len().mean()), 1),
            "median": int(group.str.split().str.len().median()),
        }
        for label, group in frame.groupby("label")["text"]
    }

    frame["_dedupe"] = frame["text"].str.lower()
    dup = frame.duplicated("_dedupe", keep="first")
    conflicts = frame[frame["_dedupe"].duplicated(keep=False)].groupby("_dedupe")["label"].nunique()
    frame = frame[~dup].drop(columns="_dedupe").reset_index(drop=True)

    truncated = int((frame["text"].str.len() > MAX_CHARS).sum())
    frame["text"] = frame["text"].map(truncate)

    frame["id"] = (
        frame["source"] + "_" + (frame.groupby("source").cumcount() + 1).astype(str).str.zfill(6)
    )
    frame["split"] = "train"
    frame = frame[["id", "text", "lang", "label", "type", "source", "split"]]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False, encoding="utf-8")
    meta = {
        "rows_raw": int(len(frame) + dup.sum()),
        "rows_csv": int(len(frame)),
        "dup_dropped": int(dup.sum()),
        "dup_label_conflicts": int((conflicts > 1).sum()),
        "rows_raw_by_source": {k: int(v) for k, v in raw_counts.items()},
        "rows_by_source": {k: int(v) for k, v in frame["source"].value_counts().items()},
        "max_chars": MAX_CHARS,
        "truncated_rows": truncated,
        "length_before_cap": length_before,
        "words_before_cap": word_counts,
        "file_bytes": OUT.stat().st_size,
    }
    META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nWrote {OUT} with {len(frame)} rows, dropped {int(dup.sum())} duplicate texts")
    print(f"duplicate groups with both labels: {int((conflicts > 1).sum())}")
    print(f"truncated to {MAX_CHARS} chars: {truncated} rows")
    print(f"size: {OUT.stat().st_size / 1e6:.1f} MB")
    print(frame["label"].value_counts().to_dict())
    print(frame["lang"].value_counts().head(10).to_dict())


if __name__ == "__main__":
    main()
