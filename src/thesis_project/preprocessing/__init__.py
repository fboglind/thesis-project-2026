"""Preprocessing module: data loading and text normalisation."""

from .data_loader import load_fas_data, load_svf_data
from .normalizer import norm

__all__ = [
    "load_fas_data",
    "load_svf_data",
    "norm",
]
