import sys

from huggingface_hub import snapshot_download

from simple_local.config import load


def main() -> None:
    config_path = sys.argv[1]
    for spec in load(config_path).models:
        model = spec.config.get("model")
        if model:
            print(f"caching {model}")
            snapshot_download(model)


if __name__ == "__main__":
    main()
