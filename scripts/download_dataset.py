#!/usr/bin/env python3
"""Download benchmark images and annotations from the Hugging Face dataset hub."""
import argparse
from huggingface_hub import snapshot_download


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-id', required=True, help='Hugging Face namespace/InspectBench')
    p.add_argument('--local-dir', default='data/InspectBench')
    p.add_argument('--revision', default='main', help='Use a commit hash to pin the dataset version')
    a = p.parse_args()
    print(snapshot_download(repo_id=a.repo_id, repo_type='dataset', revision=a.revision, local_dir=a.local_dir))


if __name__ == '__main__':
    main()
