"""Скачивание MovieLens: python scripts/download_movielens.py [ml-1m|ml-100k|ml-latest-small]"""
import sys

from grouprec.datasets import download_movielens

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "ml-1m"
    print("Готово:", download_movielens(name, "data/raw"))