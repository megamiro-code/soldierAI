# ============================================================
# Standalone local Tag training / replay
# ============================================================
# Usage:
#   python tag_game.py
#   python tag_game.py --updates 200
#   python tag_game.py --replay --policy 12 --bout 7
#
# This file performs ALL tag computation locally (CPU/JAX).
# It can optionally sync the current production Elite from the Modal
# Volume before training, then upload the newly created Tag Elite and
# replay files back to the Volume so the next `modal run main.py` can
# consume them as the starting candidate.

import argparse
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCAL_BASE_DIR = ROOT / "PPO_RTS"
os.environ["RTS_BASE_DIR"] = str(LOCAL_BASE_DIR)

import modal  # local SDK only; no Modal compute is started here

import main as rts

REMOTE_VOLUME_NAME = "rts-storage"
REMOTE_ROOT = "PPO_RTS"


def _parse_generation(name: str) -> int:
    m = re.search(r"generation_(\d+)", name)
    return int(m.group(1)) if m else -1


def _parse_tag_policy(name: str) -> int:
    m = re.search(r"tag_policy_(\d+)", name)
    return int(m.group(1)) if m else -1


def sync_remote_seed_files():
    """Download remote production/tag seeds into the local tag workspace."""
    try:
        vol = modal.Volume.from_name(REMOTE_VOLUME_NAME, create_if_missing=True)
        LOCAL_BASE_DIR.mkdir(parents=True, exist_ok=True)
        local_elite = LOCAL_BASE_DIR / "elite"
        local_tag = LOCAL_BASE_DIR / "tag_elite"
        local_elite.mkdir(exist_ok=True)
        local_tag.mkdir(exist_ok=True)

        downloaded = 0
        local_bouts = LOCAL_BASE_DIR / "tag_bouts"
        local_replay = LOCAL_BASE_DIR / "tag_replay"
        local_bouts.mkdir(exist_ok=True)
        local_replay.mkdir(exist_ok=True)

        for remote_dir, local_dir, pattern in (
            (f"{REMOTE_ROOT}/elite", local_elite, "generation_"),
            (f"{REMOTE_ROOT}/tag_elite", local_tag, "tag_policy_"),
            (f"{REMOTE_ROOT}/tag_bouts", local_bouts, "tag_policy_"),
            (f"{REMOTE_ROOT}/tag_replay", local_replay, "tag_replay_policy_"),
        ):
            try:
                entries = list(vol.listdir(remote_dir, recursive=False))
            except Exception:
                continue
            for entry in entries:
                remote_path = str(entry.path)
                name = Path(remote_path).name
                if not name.endswith(".npz") or not name.startswith(pattern):
                    continue
                # Keep the workspace manageable: production Elite and Tag Elite
                # files are normally small enough to download individually.
                data = b"".join(vol.read_file(remote_path))
                (local_dir / name).write_bytes(data)
                downloaded += 1
        print(f"Modal seed sync      : {downloaded} file(s)")
    except Exception as exc:
        print(f"Modal seed sync skipped: {exc}")


def upload_tag_results():
    """Upload local Tag Elite and replay files to the persistent Modal Volume."""
    vol = modal.Volume.from_name(REMOTE_VOLUME_NAME, create_if_missing=True)
    tag_elite = LOCAL_BASE_DIR / "tag_elite"
    tag_bouts = LOCAL_BASE_DIR / "tag_bouts"
    tag_replay = LOCAL_BASE_DIR / "tag_replay"

    files = []
    for base, remote_dir in (
        (tag_elite, f"/{REMOTE_ROOT}/tag_elite"),
        (tag_bouts, f"/{REMOTE_ROOT}/tag_bouts"),
        (tag_replay, f"/{REMOTE_ROOT}/tag_replay"),
    ):
        if not base.exists():
            continue
        for path in base.iterdir():
            if path.is_file():
                files.append((path, remote_dir))

    if not files:
        print("Modal upload          : no new Tag files")
        return

    with vol.batch_upload(force=True) as batch:
        for path, remote_dir in files:
            batch.put_file(str(path), f"{remote_dir}/{path.name}")
    print(f"Modal Tag upload      : {len(files)} file(s)")


def run_training(updates: int, sync_remote: bool, upload_remote: bool):
    if sync_remote:
        sync_remote_seed_files()

    tag_policy_path, representative_path = rts.run_tag_training(
        n_updates=updates,
        resume=True,
    )
    print()
    print("Local Tag training finished")
    print(f"Tag policy            : {tag_policy_path}")
    print(f"Representative replay : {representative_path}")
    print(f"Local replay directory: {LOCAL_BASE_DIR / 'tag_replay'}")

    if upload_remote:
        upload_tag_results()


def run_replay(policy: int | None, bout: int, sync_remote: bool):
    if sync_remote:
        sync_remote_seed_files()
    path = rts.show_tag_replay(policy_number=policy, bout_number=bout)
    print(f"Local Tag replay      : {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Local-only Tag training and replay"
    )
    parser.add_argument(
        "--updates", type=int, default=rts.TAG_UPDATES_PER_RUN,
        help=f"Tag optimizer updates (default: {rts.TAG_UPDATES_PER_RUN})",
    )
    parser.add_argument(
        "--replay", action="store_true",
        help="Replay an existing saved Tag game instead of training",
    )
    parser.add_argument(
        "--policy", type=int, default=None,
        help="Tag policy number to replay (omit for latest)",
    )
    parser.add_argument(
        "--bout", type=int, default=0,
        help="Bout number within the Tag policy (default: 0)",
    )
    parser.add_argument(
        "--sync", action="store_true",
        help="Before training/replay, download the latest Tag/production files from Modal Volume",
    )
    parser.add_argument(
        "--upload", action="store_true",
        help="After local Tag training, upload Tag policy and replays to Modal Volume",
    )
    args = parser.parse_args()

    if args.replay:
        run_replay(args.policy, args.bout, sync_remote=args.sync)
    else:
        run_training(args.updates, sync_remote=args.sync, upload_remote=args.upload)


if __name__ == "__main__":
    main()
