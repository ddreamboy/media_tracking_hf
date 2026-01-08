import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo
from loguru import logger

load_dotenv()

HF_USERNAME = os.getenv("HF_USERNAME")

MODEL_REPO = f"{HF_USERNAME}/{os.getenv('MODEL_REPO')}"
DATASET_REPO = f"{HF_USERNAME}/{os.getenv('DATASET_REPO')}"

MODEL_VERSION = "v0.1.0"
DATASET_VERSION = "2026-01-08"

MODEL_DIR = Path("model") / MODEL_VERSION
DATASET_DIR = Path("dataset")

MODEL_TAG = f"model-{MODEL_VERSION}"
DATASET_TAG = f"dataset-{DATASET_VERSION}"

api = HfApi()

create_repo(MODEL_REPO, repo_type="model", exist_ok=True)
create_repo(DATASET_REPO, repo_type="dataset", exist_ok=True)

logger.info("Uploading model files...")

for file in MODEL_DIR.iterdir():
    if not file.is_file():
        continue

    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=f"{MODEL_VERSION}/{file.name}",
        repo_id=MODEL_REPO,
        repo_type="model",
        commit_message=f"Upload {MODEL_VERSION} ({file.name})",
    )

    logger.info(f"model/{MODEL_VERSION}/{file.name}")

api.create_tag(
    repo_id=MODEL_REPO,
    tag=MODEL_TAG,
    repo_type="model",
)
logger.info(f"Model tag created: {MODEL_TAG}")

logger.info("Uploading datasets...")

for subdir in ["raw", "processed"]:
    for file in (DATASET_DIR / subdir).iterdir():
        if not file.is_file():
            continue

        api.upload_file(
            path_or_fileobj=file,
            path_in_repo=f"{subdir}/{file.name}",
            repo_id=DATASET_REPO,
            repo_type="dataset",
            commit_message=f"Update dataset ({subdir})",
        )

        logger.info(f"dataset/{subdir}/{file.name}")

api.create_tag(
    repo_id=DATASET_REPO,
    tag=DATASET_TAG,
    repo_type="dataset",
)
logger.info(f"Dataset tag created: {DATASET_TAG}")

logger.info("Upload + tagging complete")
