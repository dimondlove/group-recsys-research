"""Запуск эксперимента: python scripts/run_experiment.py configs/pilot.yaml"""
import argparse
from pathlib import Path

from grouprec.experiments.runner import main

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", help="путь к YAML-конфигу, например configs/pilot.yaml")
    ap.add_argument("--base-dir", default=".", help="корень проекта (относительно него читаются data/ и пишутся results/)")
    ap.add_argument("--seed", type=int, default=None, help="переопределить seed из конфига")
    ap.add_argument("--quiet", action="store_true", help="без индикатора прогресса")
    args = ap.parse_args()
    main(Path(args.config), args.base_dir, progress=not args.quiet, seed=args.seed)