"""Feature extraction and exploratory sensor analysis utilities."""

from .feature_analysis import (
    compute_axis_correlations,
    compute_class_feature_stats,
    compute_window_time_domain_features,
    plot_eda_summary,
)

__all__ = [
    "compute_class_feature_stats",
    "compute_axis_correlations",
    "compute_window_time_domain_features",
    "plot_eda_summary",
]
