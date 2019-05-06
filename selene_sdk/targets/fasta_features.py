"""
This class contains methods to query a fasta file.

It accepts the path to a directory containing fasta files.

Standard fasta format must be followed.
"""
import types
import os

import pyfaidx
import numpy as np

from .target import Target

class FastaFeatures(Target):
    """FastaFeatures
    Stores the dataset specifying sequences and features.
    Accepts a directory containing fasta files.

    Parameters
    ----------
    input_path : str
        Path to the tabix-indexed dataset. Note that for the file to
        be tabix-indexed, it must have been compressed with `bgzip`.
        Thus, `input_path` should be a `*.gz` file with a
        corresponding `*.tbi` file in the same directory.
    features : list(str)
        The non-redundant list of genomic features (i.e. labels)
        that will be predicted.

    Attributes
    ----------
    data : tabix.open
        The data stored in a tabix-indexed `*.bed` file.
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

    def __init__(self, input_path, features):
        """
        Constructs a new `FastaFeatures` object.
        """
        self.data = []
        fasta_files = os.listdir(input_path)
        valid_fastas = []
        for i, fasta_in in enumerate(fasta_files):
            if not '.fa' in fasta_in:
                continue
            if '.fai' in fasta_in:
                continue
            self.data.append(pyfaidx.Fasta(os.path.join(input_path,fasta_in), duplicate_action="first"))
            valid_fastas.append(fasta_in)

        self.n_features = len(features)

        self.feature_index_dict = dict(
            [(feat, index) for index, feat in enumerate(features)])

        self.index_feature_dict = dict(list(enumerate(features)))

        self.file_index_dict = dict(
            [(fasta, index) for index, fasta in enumerate(valid_fastas)])

        self.index_file_dict = dict(list(enumerate(valid_fastas)))

        self._features = features

    def get_feature_data(self, labels):
        """
        Extract the features (labels) for a sequence of interest using the name
        to query the labels provided.

        Parameters
        ----------
        labels : str
            string containing the labels for a given observation, 
            split by underscores for multiple labels.

        Returns
        -------
        numpy.ndarray
            Array of one-hot labels.

        """
        one_hot_labels = np.zeros((self.n_features))
        for feature in self._features:
            feature_vec = feature.split('|')[1].split('_')
            feature_lower = [x.upper() for x in feature_vec]
            labs_lower = [x.upper() for x in labels.split('_')]
            if any(np.in1d(labs_lower,feature_lower)):
                 one_hot_labels[self.feature_index_dict[feature]] = 1
        return one_hot_labels
