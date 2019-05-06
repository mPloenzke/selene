"""
This module contains classes and methods for target feature classes.
These are classes which define a way to access a "target feature" such
as a label or annotation on an input sequence.
"""
from .target import Target
from .genomic_features import GenomicFeatures
from .fasta_features import FastaFeatures
from .h5_features import H5Features

__all__ = ["Target", "GenomicFeatures", "FastaFeatures", "H5Features"]
