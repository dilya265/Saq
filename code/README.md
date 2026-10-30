# SAQ code

Code for the SAQ Kazakh-language scam message detector (XLM-RoBERTa fine-tuning).

- Project page: https://dilya265.github.io/Saq/
- Demo: https://dilya265.github.io/Saq/Сақ_%20хабарлама%20алаяқтық%20па_.html
- Model weights: https://huggingface.co/wiiqliw/saq-scam-xlmr

## Files

| File | What it does |
|---|---|
| `gpu/train.py` | Fine-tunes XLM-RoBERTa for `safe` / `scam` on the GPU queue |
| `gpu/predict.py` | Runs the trained model over many messages |
| `gpu/serve.py` | Small local HTTP service that scores one message |
| `predict_xlmr.py` | Predicts one message from the command line |
| `build_after.py` | Builds the training file from public data + local messages |
| `make_saq_train.py`, `make_test_split.py` | Build the local training rows and the frozen test set |
| `eval_compare.py`, `eval_variant.py` | Compare methods on the same 100-message test set |
| `freeze_test.py` | Freezes the test set and writes its md5 |
| `near_copies.py` | Checks test messages are not near-copies of training messages |
| `scripts/build_data.py` | Builds the public message dataset |
| `scripts/saq_check.py` | Rule-based keyword baseline |
| `scripts/make_report.py` | Makes the data report and plots |

## Model

Do not put the weights in GitHub (1.1 GB). Download them from Hugging Face:

```bash
mkdir -p models
hf download wiiqliw/saq-scam-xlmr --local-dir models/scam-xlmr
```

Then:

```bash
python3 predict_xlmr.py models/scam-xlmr "Служба безопасности Kaspi: назовите код из SMS"
```

Training needs the data files (`data/messages.csv`, `data/saq_train.csv`) and a GPU:
submit `gpu/train.py` as a GPU job (see the class GPU guide), not on the login node.

## Results

Public check set (7623 messages): recall 97.6%, precision 99.0%, 26 false alarms out of 5014 safe.
Frozen local test (100 messages): 49/50 scams caught, 2/50 false alarms, 97% accuracy.

The data files are not in this repository. The test set was never used for training.
