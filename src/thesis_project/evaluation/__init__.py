"""Evaluation primitives for the thesis project.

Re-exports the SQ3 public surface so callers can do
``from thesis_project.evaluation import compute_reliability`` etc.
"""

from .sq3_agreement import (
    AgreementResult,
    SaldoAgreementResult,
    rater_model_per_category,
    rater_model_per_quartile,
    rater_model_spearman,
    rater_saldo_spearman,
)
from .sq3_divergence import compute_divergence_catalog
from .sq3_reliability import (
    VALID_CATEGORIES,
    VALID_RATINGS,
    ReliabilityReport,
    compute_reliability,
)
from .sq3_sampling import (
    PRIMARY_SEED,
    RATER_COLUMNS,
    SENSITIVE_COLUMNS,
    TRAINING_SEED,
    assign_quartile,
    compute_quartile_cutpoints,
    draw_stratified_sample,
    draw_training_sample,
    load_eligible_pairs,
    make_rater_csv,
)

__all__ = [
    "AgreementResult",
    "PRIMARY_SEED",
    "RATER_COLUMNS",
    "ReliabilityReport",
    "SENSITIVE_COLUMNS",
    "SaldoAgreementResult",
    "TRAINING_SEED",
    "VALID_CATEGORIES",
    "VALID_RATINGS",
    "assign_quartile",
    "compute_divergence_catalog",
    "compute_quartile_cutpoints",
    "compute_reliability",
    "draw_stratified_sample",
    "draw_training_sample",
    "load_eligible_pairs",
    "make_rater_csv",
    "rater_model_per_category",
    "rater_model_per_quartile",
    "rater_model_spearman",
    "rater_saldo_spearman",
]
