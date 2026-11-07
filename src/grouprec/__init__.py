"""grouprec: агрегация предпочтений для групповых рекомендаций и анализа коллективного выбора."""
from .compare import compare_rules
from .data import (TASK_LIMITS, InputError, Limits, load_ratings_csv, parse_rankings,
                   top_n_union, validate_ratings)
from .metrics import (fairness_summary, gini, group_cohesion, member_satisfaction,
                      topk_overlap)
from .rules import (AggregationRule, available_rules, get_rule, pairwise_preferences,
                    rank_items, register, top_ties, unregister)
from .synthetic import make_group_ratings

__all__ = [
    "AggregationRule", "InputError", "Limits", "TASK_LIMITS", "available_rules", "compare_rules",
    "fairness_summary", "get_rule", "gini", "group_cohesion", "load_ratings_csv",
    "make_group_ratings", "member_satisfaction", "pairwise_preferences", "parse_rankings",
    "rank_items", "register", "top_n_union", "top_ties", "topk_overlap", "unregister", "validate_ratings",
]