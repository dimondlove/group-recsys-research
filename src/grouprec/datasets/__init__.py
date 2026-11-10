from .movielens import Ratings, download_movielens, load_movielens
from .split import split_ratings

# Фиктивные данные подключаются явно: from grouprec.datasets.fake import make_fake_movielens
__all__ = ["Ratings", "download_movielens", "load_movielens", "split_ratings"]