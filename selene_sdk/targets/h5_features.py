"""
This class contains methods for input labels used in the h5 sampler..

It accepts a N-by-P matrix for P labels.
"""
import types
import os

import numpy as np

from .target import Target

class H5Features(Target):
    """H5Features
    Stores the dataset specifying features.

    Parameters
    ----------
    features : list(str)
        The non-redundant list of genomic features (i.e. labels)
        that will be predicted as a N-by-P matrix for P labels.

    Attributes
    ----------
    n_features : int
        The number of distinct features.
    feature_index_dict : dict
        A dictionary mapping feature names (`str`) to indices (`int`),
        where the index is the position of the feature in `features`.
    index_feature_dict : dict
        A dictionary mapping indices (`int`) to feature names (`str`),
        where the index is the position of the feature in the input
        features.
    """

    def __init__(self, features):
        """
        Constructs a new `FastaFeatures` object.
        """
        self.n_features = len(features)

        self.feature_index_dict = dict(
            [(feat, index) for index, feat in enumerate(features)])

        self.index_feature_dict = dict(list(enumerate(features)))

        self._features = features
    def get_feature_data(self, labels):
        """
        Helper to return the labels since they are already one-hot encoded.

        """
        return labels
