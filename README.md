# SAQ — Kazakh-Language Scam Message Detection

Student project at **ISSAI FRP** by **Assem** and **Akzhan**.

SAQ reads a suspicious message and says whether it looks like a scam, and why.
It is made for Kazakh-speaking users in Kazakhstan, especially older adults.

- **Project page:** https://dilya265.github.io/Saq/
- **Demo (Kazakh):** https://dilya265.github.io/Saq/Сақ_%20хабарлама%20алаяқтық%20па_.html
- **Model:** https://huggingface.co/wiiqliw/saq-scam-xlmr
- **Code:** [code/](code/)

## Model

Fine-tuned XLM-RoBERTa (`safe` / `scam`).

| Check | Result |
|---|---|
| Public check set, 7623 messages | recall 97.6%, precision 99.0%, 26 false alarms / 5014 safe |
| Frozen local test, 100 messages | 49/50 scams caught, 2/50 false alarms, 97% accuracy |
| Kazakh messages on the public check | 219 / 219 correct |
| Russian messages on the public check | 55 / 55 correct |

The model weights (1.1 GB) live on Hugging Face:

```bash
hf download wiiqliw/saq-scam-xlmr --local-dir models/scam-xlmr
```

## Run

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install torch transformers
python3 code/predict_xlmr.py models/scam-xlmr "Служба безопасности Kaspi: назовите код из SMS"
```

Training and evaluation scripts are in [code/](code/). Training runs on a GPU job.

## Limitations

- The local test is small: 100 messages, so 1–2 messages are within noise.
- Safe examples are mostly normal chat messages; service messages with real codes are rare.
- The tool gives a hint, not a verdict. A person should still decide.
