"""The live check page (CPU, no GPU needed: one message takes well under a second).
  python3 gpu/serve.py        then open https://kitan-a.com/frp/s01/proxy/8000/  (keep the terminal open)
POST /check {"text": "..."} -> {"label": "scam"|"safe", "scam_prob": 0.97}. Never log the messages people paste.
"""
from pathlib import Path

import torch
from flask import Flask, jsonify, request
from transformers import AutoModelForSequenceClassification, AutoTokenizer

M = Path(__file__).resolve().parent.parent / "models/scam-xlmr"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForSequenceClassification.from_pretrained(M).eval()
torch.set_num_threads(4)
app = Flask(__name__)

PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Scam check</title><body style="font:17px system-ui;max-width:640px;margin:30px auto;padding:0 16px">
<h2>Is this message a scam?</h2><textarea id=t rows=6 style="width:100%;font:inherit"></textarea>
<button onclick="go()" style="font:inherit;padding:8px 18px;margin-top:8px">Check</button><h3 id=r></h3>
<script>async function go(){const r=await fetch('check',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({text:document.getElementById('t').value})});const j=await r.json();
document.getElementById('r').textContent=(j.label==='scam'?'⚠️ SCAM':'✅ looks safe')+'  ('+Math.round(j.scam_prob*100)+'%)';}</script>"""


@app.get("/")
def page():
    return PAGE


@app.post("/check")
def check():
    text = str((request.get_json(silent=True) or {}).get("text", ""))[:2000]
    with torch.no_grad():
        p = torch.softmax(model(**tok(text, truncation=True, max_length=128, return_tensors="pt")).logits, -1)[0, 1].item()
    return jsonify(label="scam" if p >= .5 else "safe", scam_prob=round(p, 4))


app.run(host="127.0.0.1", port=8000)
