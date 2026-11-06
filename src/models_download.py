"""Pre-fetch the pinned model snapshots into the HF cache (`make models`)."""

from huggingface_hub import snapshot_download

from src.config import MODELS

if __name__ == "__main__":
    for name, (repo, rev) in MODELS.items():
        path = snapshot_download(
            repo, revision=rev,
            # Skip duplicate weight formats (onnx/openvino/tf/flax) we never load.
            allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "merges.txt",
                            "vocab.*", "1_Pooling/*", "modules.json", "sentence_*"],
        )
        print(f"{name:7s} {repo}@{rev[:10]} -> {path}")
