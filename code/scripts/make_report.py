"""Write data_report.md and the plot from data/messages.csv.

Run: python3 scripts/make_report.py
"""

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "messages.csv"
META = ROOT / "data" / "messages.meta.json"
PLOT = ROOT / "data" / "plot_label_by_source.png"
REPORT = ROOT / "data_report.md"

SOURCE_LABELS = {
    "sms_spam": "ucirvine/sms_spam",
    "difraud_sms": "redasers/difraud: sms",
    "difraud_phishing": "redasers/difraud: phishing",
    "difraud_job_scams": "redasers/difraud: job_scams",
    "fz_junkmail": "FredZhang7/all-scam-spam",
}

STOPWORDS = set(
    """the and to of a in is that for it you on with this be are as at your or have not from by we can will if an
    was were but they their there what when who how why all any some more most other into than then them these those
    our out up do does did done has had been being just about over after before under again very also its i me my he
    she his her him us them s t don now get got would could should may might must shall im ive id youre
    и в не на что с по это как к у из за от для до же бы ли то но а или так его она они мы вы я ты он
    http https www com html br nbsp amp quot gt lt mailto href font http equot""".split()
)

WORD = re.compile(r"[^\W\d_]{2,}", re.UNICODE)


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def text_snippet(text, limit=220):
    text = text.replace("|", "/")
    return text[:limit] + ("..." if len(text) > limit else "")


def draw_plot(frame, meta):
    sources = [s for s in SOURCE_LABELS if s in set(frame["source"])]
    counts = frame.pivot_table(index="source", columns="label", values="id", aggfunc="count").fillna(0)
    counts = counts.reindex(sources)

    width, height = 1400, 620
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 20)
    small = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 16)
    bold = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSansBold.ttf", 26)

    draw.text((width // 2 - 330, 20), "Scam vs safe messages per public dataset", font=bold, fill="black")

    left, top, bottom = 90, 130, 470
    chart_bottom = bottom
    max_count = int(np.ceil(counts.to_numpy().max() / 5000) * 5000)
    plot_width = width - left - 60
    group_width = plot_width / len(sources)
    bar_width = group_width * 0.32

    for step in range(6):
        y = chart_bottom - (chart_bottom - top) * step / 5
        value = int(max_count * step / 5)
        draw.line((left, y, width - 60, y), fill="#dddddd")
        draw.text((10, y - 10), f"{value:,}", font=small, fill="#555555")

    colors = {"scam": "#d64541", "safe": "#3f8f5f"}
    for index, source in enumerate(sources):
        center = left + group_width * (index + 0.5)
        for offset, label in [(-0.18, "scam"), (0.18, "safe")]:
            value = int(counts.loc[source, label])
            bar_height = (chart_bottom - top) * value / max_count
            x0 = center + group_width * offset - bar_width / 2
            draw.rectangle(
                (x0, chart_bottom - bar_height, x0 + bar_width, chart_bottom),
                fill=colors[label],
            )
            draw.text(
                (x0 + bar_width / 2 - 18, chart_bottom - bar_height - 24),
                f"{value:,}",
                font=small,
                fill="#333333",
            )
        name = SOURCE_LABELS[source].replace("redasers/difraud: ", "difraud\n").replace("ucirvine/", "").replace(
            "FredZhang7/", ""
        )
        for line_index, line in enumerate(name.split("\n")):
            text_width = draw.textlength(line, font=small)
            draw.text((center - text_width / 2, chart_bottom + 12 + line_index * 20), line, font=small, fill="black")

    draw.rectangle((width // 2 - 110, 78, width // 2 - 90, 98), fill=colors["scam"])
    draw.text((width // 2 - 80, 76), "scam", font=small, fill="black")
    draw.rectangle((width // 2 + 30, 78, width // 2 + 50, 98), fill=colors["safe"])
    draw.text((width // 2 + 60, 76), "safe", font=small, fill="black")

    caption = (
        f"{len(frame):,} messages kept of {meta['rows_raw']:,} raw rows; "
        f"{meta['dup_dropped']:,} duplicate texts removed"
    )
    draw.text((left, height - 40), caption, font=small, fill="#555555")
    image.save(PLOT)


def main():
    frame = pd.read_csv(CSV)
    meta = json.loads(META.read_text(encoding="utf-8"))
    draw_plot(frame, meta)

    rows = []
    for source, label in SOURCE_LABELS.items():
        part = frame[frame["source"] == source]
        scam = int((part["label"] == "scam").sum())
        safe = int((part["label"] == "safe").sum())
        raw = meta["rows_raw_by_source"].get(source, 0)
        rows.append(
            [
                label,
                f"{raw:,}",
                f"{len(part):,}",
                f"{raw - len(part):,}",
                f"{scam:,}",
                f"{safe:,}",
                f"{100 * scam / len(part):.1f}%" if len(part) else "-",
                int(part["text"].str.len().median()) if len(part) else 0,
            ]
        )
    per_source = md_table(
        ["Dataset", "raw rows", "in CSV", "dupes dropped", "scam", "safe", "scam %", "median chars"], rows
    )

    total = len(frame)
    scam_total = int((frame["label"] == "scam").sum())
    safe_total = total - scam_total

    lang_counts = frame["lang"].value_counts()
    lang_rows = [[code, f"{count:,}", f"{100 * count / total:.2f}%"] for code, count in lang_counts.items()]

    lang_by_source = []
    for source, label in SOURCE_LABELS.items():
        part = frame[frame["source"] == source]
        counts = part["lang"].value_counts()
        top = ", ".join(f"{code} {100 * count / len(part):.1f}%" for code, count in counts.head(4).items())
        lang_by_source.append([label, top])

    length_rows = []
    for label in ["scam", "safe"]:
        before = meta["length_before_cap"][label]
        words = meta["words_before_cap"][label]
        length_rows.append(
            [
                label,
                before["mean"],
                before["median"],
                before["p90"],
                before["p99"],
                f"{before['max']:,}",
                words["mean"],
                words["median"],
            ]
        )

    def top_words(subset, limit):
        counter = Counter()
        for text in subset:
            counter.update(WORD.findall(text.lower()))
        return counter, counter.most_common(limit)

    scam_words, scam_top = top_words(frame.loc[frame["label"] == "scam", "text"], 20)
    safe_words, safe_top = top_words(frame.loc[frame["label"] == "safe", "text"], 20)

    def content_top(counter, limit=15):
        items = [(w, c) for w, c in counter.most_common(400) if w not in STOPWORDS]
        return items[:limit]

    scam_content = content_top(scam_words)
    safe_content = content_top(safe_words)

    examples = []
    for label in ["scam", "safe"]:
        part = frame[frame["label"] == label]
        picks = part.groupby("source", group_keys=False).head(2).head(10)
        for _, row in picks.iterrows():
            examples.append(f"**{label}** · `{row['source']}` · {row['lang']}\n> {text_snippet(row['text'])}")

    kk_rows = int((frame["lang"] == "kk").sum())
    ru_rows = int((frame["lang"] == "ru").sum())
    ru_picks = frame[frame["lang"] == "ru"].head(2)
    ru_examples = "\n".join(
        f"> **{row['label']}** · `{row['source']}` — {text_snippet(row['text'], 160)}"
        for _, row in ru_picks.iterrows()
    )

    report = f"""# Data report: public scam datasets

Built by `scripts/build_data.py` and `scripts/make_report.py` from `data/messages.csv`
({total:,} messages, {meta['file_bytes'] / 1e6:.1f} MB). All rows have `split = train`; the real test set will come
later from `data/local.csv`.

## 1. Messages per dataset

{per_source}

**Total: {total:,} messages — {scam_total:,} scam ({100 * scam_total / total:.1f}%) and {safe_total:,} safe.**
{meta['dup_dropped']:,} duplicate texts were removed before writing the CSV ({meta['dup_label_conflicts']} pairs had
conflicting labels in different sources and the first one was kept).

## 2. Scam against safe

| label | messages | share |
|---|---|---|
| scam | {scam_total:,} | {100 * scam_total / total:.1f}% |
| safe | {safe_total:,} | {100 * safe_total / total:.1f}% |

The data is imbalanced: about one scam for every two safe messages. The scam share per dataset runs from
{100 * (frame[(frame['source'] == 'difraud_job_scams')]['label'] == 'scam').mean():.1f}% (difraud job_scams) to
{100 * (frame[(frame['source'] == 'fz_junkmail')]['label'] == 'scam').mean():.1f}% (FredZhang7/all-scam-spam), so the
five sets disagree a lot about what "scam" means.

## 3. Languages

Estimated with a script + stopword heuristic in `build_data.py` (not a real language detector, see limits).

{md_table(["lang", "messages", "share"], lang_rows)}

Per dataset (top 4):

{md_table(["Dataset", "languages"], lang_by_source)}

**The important result for our project: Russian is only {ru_rows} rows ({100 * ru_rows / total:.2f}%) and Kazakh is
{kk_rows} rows (0%).** The public data is ~99% English, so it cannot teach the detector Russian or Kazakh scam wording.
That is exactly why the 120 real local messages are needed.

The {ru_rows} Russian rows are all short sentences (median {int(frame[frame['lang'] == 'ru']['text'].str.len().median())} characters,
{int((frame[frame['lang'] == 'ru']['label'] == 'scam').sum())} scam / {int((frame[frame['lang'] == 'ru']['label'] == 'safe').sum())} safe) and the "scams" are generic ads
("Получите кредит с минимальной процентной ставкой!"), not real Kazakhstani bank, Kaspi or delivery messages:

{ru_examples}

## 4. Message length

Length before the 2,000-character cap (the CSV itself is capped, see limits):

{md_table(["label", "mean chars", "median", "p90", "p99", "max", "mean words", "median words"], length_rows)}

Median message length is short ({int(meta['length_before_cap']['scam']['median'])} chars for scams,
{int(meta['length_before_cap']['safe']['median'])} for safe), but e-mails pull the mean up a lot:
{meta['truncated_rows']:,} rows ({100 * meta['truncated_rows'] / total:.1f}%) were longer than 2,000 characters and
were cut.

## 5. The 20 most common words per class

Raw tokens (letters only, lowercased, 2+ characters; HTML and quoted-printable debris is NOT removed):

**Scam:** {", ".join(f"`{w}` ({c:,})" for w, c in scam_top)}

**Safe:** {", ".join(f"`{w}` ({c:,})" for w, c in safe_top)}

Without the most common English/Russian function words:

**Scam:** {", ".join(f"`{w}` ({c:,})" for w, c in scam_content)}

**Safe:** {", ".join(f"`{w}` ({c:,})" for w, c in safe_content)}

## 6. Plot

![scam vs safe per dataset](data/plot_label_by_source.png)

## 7. Ten examples per class

{chr(10).join(example + chr(10) for example in examples)}

## 8. How scams differ (one line)

Scams carry money and urgency words (account, click, free, offer, money, credit, prize, your) while safe messages are
ordinary conversation, so a keyword/TF-IDF baseline should work in English — but almost all of this vocabulary is
English, and the Russian/Kazakh scam words are missing from the public data.

## 9. Limits and problems found

- **Language label is an estimate.** Cyrillic is split into `ru` and `kk` by Kazakh letters (ә, ғ, қ, ң, ө, ұ, ү, һ, і);
  Latin text defaults to `en` unless 2+ words clearly match Spanish/Italian/French/German/Portuguese/Turkish/etc.
  So non-English Latin messages are under-counted, and Kazakh written in Latin letters counts as `en`.
- **`lang = other`** ({int((frame['lang'] == 'other').sum()):,} rows) means non-Latin script (Arabic, CJK, Greek...) or
  no letters at all (e.g. "645", ":-)").
- **Encoding is broken in the sources.** Many e-mails contain `=20` (quoted-printable), HTML tags, or `���`
  replacement characters, mostly in `FredZhang7/all-scam-spam` and `redasers/difraud`. We did not repair the text; it
  should be cleaned before training.
- **`text` is capped at 2,000 characters** in the CSV (same length as a 512-token model input) so the file stays at
  {meta['file_bytes'] / 1e6:.1f} MB and can be pushed to GitHub. The raw downloads in `data/raw/` keep the full text.
- **Type is coarse.** `difraud` does not label scam types, so `type` comes from the subset: sms → `other`,
  phishing → `phishing_link`, job_scams → `job_offer`, `all-scam-spam` and `sms_spam` → `other`. Safe rows use `none`.
- **`redasers/difraud: sms` repeats itself: 4,752 of its 6,574 rows are copies of another row.** The same messages
  appear in its train, validation and test files, so its own test split would leak; we deduplicated, keep all its rows
  in `train`, and never use it for measuring.
- **No `split = test` here.** These are public training rows only; measuring on them would be cheating.

## 10. How to rebuild

From the project folder:

```
hf download FredZhang7/all-scam-spam --repo-type dataset --include "junkmail_dataset.csv" --local-dir data/raw/fz
hf download redasers/difraud --repo-type dataset --include "sms/*" "phishing/*" "job_scams/*" --local-dir data/raw/difraud
hf download ucirvine/sms_spam --repo-type dataset --local-dir data/raw/sms_spam
python3 scripts/build_data.py
python3 scripts/make_report.py
```

`data/raw/` is not meant for git (110 MB); the CSV and the report are.
"""
    REPORT.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT} and {PLOT}")
    print(f"scam {scam_total:,} / safe {safe_total:,}, ru {ru_rows}, kk {kk_rows}, other {int((frame['lang'] == 'other').sum())}")


if __name__ == "__main__":
    main()
