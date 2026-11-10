"""
Разбор запуска.
  python scripts/analyze_run.py                         # последний запуск в results/runs
  python scripts/analyze_run.py results/runs/<папка>    # конкретный запуск
  python scripts/analyze_run.py --reference borda       # другое эталонное правило
"""
import argparse
import sys
from pathlib import Path

from grouprec.analysis import analyze_run, print_analysis


def list_runs(runs_dir: Path) -> list[Path]:
    """Папки запусков (в которых есть results.csv), от старых к новым."""
    if not runs_dir.is_dir():
        return []
    return sorted((p for p in runs_dir.iterdir() if (p / "results.csv").exists()),
                  key=lambda p: p.stat().st_mtime)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", default=None, help="папка запуска (по умолчанию последний запуск)")
    ap.add_argument("--runs-dir", default="results/runs", help="где искать запуски")
    ap.add_argument("--reference", default=None, help="эталонное правило (по умолчанию из конфига запуска)")
    ap.add_argument("--margin", type=float, default=0.01,
                    help="допуск эквивалентности для ndcg_mean (абсолютный); обоснуйте его до анализа")
    args = ap.parse_args()

    runs = list_runs(Path(args.runs_dir))
    if args.run_dir is None:
        if not runs:
            sys.exit(f"В {args.runs_dir} нет запусков. Сначала: python scripts/run_experiment.py configs/main.yaml")
        run_dir = runs[-1]
        print(f"Анализирую последний запуск: {run_dir}")
    else:
        run_dir = Path(args.run_dir)
        if not (run_dir / "results.csv").exists():
            msg = f"В папке {run_dir} нет results.csv."
            if runs:
                msg += "\nДоступные запуски (старые сверху):\n  " + "\n  ".join(str(p) for p in runs)
            sys.exit(msg)
    print_analysis(analyze_run(run_dir, args.reference, margin=args.margin))