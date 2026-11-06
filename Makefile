# Run everything inside the dev container. `make all` regenerates every cache
# and figure from scratch; targets with caches present are near-instant.
PY := python

.PHONY: all lock models coverage data features train experiments app examples space test

all: data features train experiments

# torch is a +cpu build: the lock is only installable with the CPU index.
lock:
	{ echo '--extra-index-url https://download.pytorch.org/whl/cpu'; pip freeze --exclude-editable | sort; } > requirements.lock.txt

models:
	$(PY) -m src.models_download

coverage:
	$(PY) experiments/00_raid_coverage.py

data:
	$(PY) -c "from src.data import build_corpus; build_corpus()"

# Hours-long, resumable: interrupted runs continue from the last finished chunk.
features:
	$(PY) -m src.features

# Fit TF-IDF / embeddings / shallow-cue LRs on train and score every non-train
# text with all detectors (artifacts/models/, artifacts/scores/). Needs features.
train:
	$(PY) -c "from src.scoring import compute_scores; compute_scores(256); compute_scores(128)"

experiments:
	$(PY) experiments/02_in_and_out_of_domain.py --tokens 256
	$(PY) experiments/02_in_and_out_of_domain.py --tokens 128
	$(PY) experiments/03_disagreement.py
	$(PY) experiments/04_homoglyph_attack.py
	$(PY) experiments/05_threshold_base_rate.py
	$(PY) app/build_examples.py

app:
	$(PY) app/app.py

# Prefilled app texts, picked from RAID by their cached scores (needs experiment 2).
examples:
	$(PY) app/build_examples.py

# Everything the Hugging Face Space needs, and nothing else, in build/space/.
# Push that folder's contents to the Space repo. Files are copied over an
# existing build/space: if SPACE_FILES ever shrinks, start from an empty folder.
SPACE_FILES := app/app.py app/examples.json \
	src/__init__.py src/config.py src/text.py src/attacks.py src/detectors.py src/metrics.py src/plotting.py \
	artifacts/calibration/calibration_256.json artifacts/calibration/calibration_128.json \
	artifacts/models/tfidf-lr_256.joblib artifacts/models/embedding-lr_256.joblib

space:
	for f in $(SPACE_FILES); do mkdir -p build/space/$$(dirname $$f) && cp $$f build/space/$$f; done
	cp app/requirements.txt build/space/requirements.txt
	cp app/space_README.md build/space/README.md
	@echo "build/space ready: $$(du -sh build/space | cut -f1)"

test:
	$(PY) -m pytest -q tests
