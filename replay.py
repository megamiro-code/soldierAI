# ============================================================
# replay.py
# Best Bout を読み込み、HTMLリプレイファイルを生成するだけのスクリプト。
# 学習は行わない（main.py を import しても train() は走らない）。
# ============================================================

import argparse
import glob
import os
import re

from main import show_replay, BOUT_DIR


# Best Bout 専用の世代番号解析。
# main.py の generation_number() は
#   generation_0035_best_bout.npz
# のような Best Bout ファイル名には対応していないため、
# replay.py 側では専用の正規表現で安全に世代番号だけを取得する。
def best_bout_generation_number(path):
    name = os.path.basename(path)
    match = re.fullmatch(r"generation_(\d+)_best_bout\.npz", name)
    if match is None:
        return -1
    return int(match.group(1))


def list_best_bout_files():
    files = glob.glob(
        os.path.join(BOUT_DIR, "generation_*_best_bout.npz")
    )
    return sorted(files, key=best_bout_generation_number)


def list_available_generations():
    files = list_best_bout_files()
    return [best_bout_generation_number(f) for f in files]


def main():
    parser = argparse.ArgumentParser(
        description="Generate an HTML replay from a saved Best Bout."
    )
    parser.add_argument(
        "--generation", "-g",
        type=int,
        default=None,
        help="World number to replay. Omit for the latest saved generation.",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all generations that have a saved Best Bout and exit.",
    )
    args = parser.parse_args()

    available = list_available_generations()

    if args.list:
        if not available:
            print(f"No Best Bout files found in {os.path.abspath(BOUT_DIR)}")
        else:
            print("Available generations with a saved Best Bout:")
            for g in available:
                print(f"  - {g}")
        return

    if not available:
        print(f"No Best Bout files found in {os.path.abspath(BOUT_DIR)}")
        print("Run training first (python main.py) until you see 'Best Bout : SAVED'.")
        return

    if args.generation is not None and args.generation not in available:
        print(f"Generation {args.generation} has no saved Best Bout.")
        print("Available generations:", available)
        return

    path = show_replay(args.generation)

    print()
    print("Replay HTML written to:")
    print(f"  {os.path.abspath(path)}")
    print()
    print("Open this file in a browser (download it, or serve it), for example:")
    print(f"  python -m http.server 8000 --directory {os.path.dirname(os.path.abspath(path))}")
    print("  then open http://localhost:8000/" + os.path.basename(path))


if __name__ == "__main__":
    main()
