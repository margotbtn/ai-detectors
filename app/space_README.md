---
title: Can you detect AI text?
emoji: 🔍
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 5.50.0
python_version: "3.12"
app_file: app/app.py
pinned: false
short_description: Three AI-text detectors, one attack, one base-rate check.
---

# Can you detect AI-written text?

Live demo for a talk on AI-text detectors. Three detectors, trained or calibrated
on RAID news (ChatGPT vs human):

- **Perplexity**: mean token log-likelihood under Qwen2.5-0.5B (zero-shot)
- **TF-IDF + logistic regression**: word and character n-grams
- **Embeddings + logistic regression**: frozen all-MiniLM-L6-v2

Tabs: score a text, attack it with look-alike letters, and see what share of
flagged texts is really AI at a given base rate.

Text is processed in memory only: nothing is logged or stored.

Code, experiments and caveats: https://github.com/margotbtn/ai-detectors
