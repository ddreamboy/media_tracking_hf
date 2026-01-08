import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download
from loguru import logger
from packaging.version import InvalidVersion, Version

load_dotenv()

HF_USERNAME = os.getenv("HF_USERNAME")

MODEL_REPO = f"{HF_USERNAME}/{os.getenv('MODEL_REPO')}"
DATASET_REPO = f"{HF_USERNAME}/{os.getenv('DATASET_REPO')}"

api = HfApi()


def get_latest_tag(repo_id: str, repo_type: str, prefix: str) -> str:
    refs = api.list_repo_refs(repo_id=repo_id, repo_type=repo_type)

    semver_tags = []
    fallback_tags = []

    for tag in refs.tags:
        if not tag.name.startswith(prefix):
            continue

        raw = tag.name[len(prefix) :]

        try:
            semver_tags.append((Version(raw.lstrip("v")), tag.name))
        except InvalidVersion:
            fallback_tags.append(tag.name)

    if semver_tags:
        semver_tags.sort(key=lambda x: x[0])
        return semver_tags[-1][1]

    if fallback_tags:
        fallback_tags.sort()
        return fallback_tags[-1]

    logger.error(f"No tags with prefix '{prefix}' found in {repo_id}")
    exit(1)


latest_model_tag = get_latest_tag(
    repo_id=MODEL_REPO,
    repo_type="model",
    prefix="model-v",
)

MODEL_VERSION = latest_model_tag.replace("model-", "")
MODEL_DIR = Path("model")

logger.info(f"Downloading model: {latest_model_tag}")

snapshot_download(
    repo_id=MODEL_REPO,
    repo_type="model",
    revision=latest_model_tag,
    local_dir=MODEL_DIR,
)

logger.info(f"Model downloaded to {MODEL_DIR}")

latest_dataset_tag = get_latest_tag(
    repo_id=DATASET_REPO,
    repo_type="dataset",
    prefix="dataset-",
)

DATASET_VERSION = latest_dataset_tag.replace("dataset-", "")
DATASET_DIR = Path("dataset")

logger.info(f"Downloading dataset: {latest_dataset_tag}")

snapshot_download(
    repo_id=DATASET_REPO,
    repo_type="dataset",
    revision=latest_dataset_tag,
    local_dir=DATASET_DIR,
)

logger.info(f"Dataset downloaded to {DATASET_DIR}")

logger.info("Download complete")
