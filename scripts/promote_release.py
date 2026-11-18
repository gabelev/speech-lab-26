"""Make a staged release branch the content of ``main`` and tag it.

    python scripts/promote_release.py GenerationGenerative/daic-woz-embeddings --branch v2 --tag v2.0

``run_pipeline.py`` with ``output.hf_revision`` uploads a release to a branch, so ``main``
keeps serving the previous version during a long run. This script then writes one commit
on ``main`` that makes it identical to the branch:
- large (LFS/xet) files are copied server side, so nothing is uploaded again;
- small files are re-uploaded from the branch;
- files on ``main`` that are not on the branch are deleted.
It then tags that commit. Earlier versions stay reachable through their tags.
"""

from __future__ import annotations

import argparse

from huggingface_hub import (CommitOperationAdd, CommitOperationCopy, CommitOperationDelete, HfApi,
                             hf_hub_download)


def tree(api: HfApi, repo: str, revision: str) -> dict:
    return {f.path: f for f in api.list_repo_tree(repo, repo_type="dataset", revision=revision, recursive=True)
            if hasattr(f, "size")}  # files only, not folders


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--branch", required=True, help="staged release branch, e.g. v2")
    ap.add_argument("--tag", required=True, help="version tag for the new main, e.g. v2.0")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    api = HfApi()
    assert api.repo_info(args.repo, repo_type="dataset").private, "refusing to touch a public repo"
    src_sha = api.repo_info(args.repo, repo_type="dataset", revision=args.branch).sha
    src, dst = tree(api, args.repo, src_sha), tree(api, args.repo, "main")

    ops, n_copy, n_add = [], 0, 0
    for path, f in sorted(src.items()):
        d = dst.get(path)
        if d is not None and getattr(d, "blob_id", None) == f.blob_id:
            continue  # unchanged
        if f.lfs is not None:
            ops.append(CommitOperationCopy(src_path_in_repo=path, path_in_repo=path, src_revision=src_sha))
            n_copy += 1
        else:
            local = hf_hub_download(args.repo, path, repo_type="dataset", revision=src_sha)
            ops.append(CommitOperationAdd(path_in_repo=path, path_or_fileobj=local))
            n_add += 1
    stale = sorted(set(dst) - set(src))
    ops += [CommitOperationDelete(path_in_repo=p) for p in stale]
    print(f"{args.branch}@{src_sha[:8]} -> main: copy {n_copy}, upload {n_add}, delete {len(stale)}, "
          f"unchanged {len(src) - n_copy - n_add}")
    if args.dry_run:
        return
    if ops:
        info = api.create_commit(args.repo, repo_type="dataset", operations=ops, revision="main",
                                 commit_message=f"Release {args.tag} (from branch {args.branch})")
        print("main is now", info.oid)
    api.create_tag(args.repo, repo_type="dataset", tag=args.tag, revision="main",
                   tag_message=f"Dataset release {args.tag}")
    print(f"tagged {args.tag}")


if __name__ == "__main__":
    main()
