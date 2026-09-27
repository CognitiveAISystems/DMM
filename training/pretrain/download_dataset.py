import argparse
import os
import re
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

DEFAULT_REPO_ID = "aandreychuk/LC-MAPF"


def download_dataset(scenario_type: str, minimal: bool, local_dir: Path,
                     repo_id: str = DEFAULT_REPO_ID):
    all_files = list_repo_files(repo_id=repo_id, repo_type="dataset")

    pattern = re.compile(rf"^{scenario_type}/part_(\d+)_(\d+)\.arrow$")
    matched_files = sorted([f for f in all_files if pattern.match(f)])
    if not matched_files:
        raise RuntimeError(f"No Arrow shards found for {scenario_type} in {repo_id}")

    if minimal:
        matched_files = matched_files[:1]  # only download 1 file if minimal

    for file in matched_files:
        hf_hub_download(
            repo_id=repo_id,
            repo_type='dataset',
            subfolder=scenario_type,
            filename=os.path.basename(file),
            local_dir=str(local_dir)
        )

def main():
    parser = argparse.ArgumentParser(description="Download the DMM pretraining Arrow dataset from Hugging Face.")
    parser.add_argument('--repo-id', default=DEFAULT_REPO_ID,
                        help='Hugging Face dataset repository ID')
    parser.add_argument('--minimal', action='store_true', help='Download only a minimal subset for local testing')
    parser.add_argument('--local_dir', type=str, default='training/pretrain/data', help='Directory to store the downloaded files')
    args = parser.parse_args()

    for scenario in ['mazes', 'house', 'random']:
        download_dataset(scenario_type=scenario, minimal=args.minimal,
                         local_dir=Path(args.local_dir) / 'train', repo_id=args.repo_id)

if __name__ == "__main__":
    main()
