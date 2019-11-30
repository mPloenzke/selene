"""
This module provides the `OnlineSampler` class and supporting methods.
Objects of the class `OnlineSampler`, are samplers which load examples
"on the fly" rather than storing them all persistently in memory.

"""
from abc import ABCMeta
import os
import random

import numpy as np

from .sampler import Sampler
from ..sequences import sequence
from ..targets import H5Features

class OnlineH5Sampler(Sampler, metaclass=ABCMeta):
    """
    A sampler in which training/validation/test data is constructed
    from random sampling of the dataset for each batch passed to the
    model. This form of sampling may alleviate the problem of loading an
    extremely large dataset into memory when developing a new model.

    Parameters
    ----------
    file_path : str
        Path to H5 file with datasets stored.
    features : list(str)
        List of distinct features that we aim to predict.
    weights : bool (optional)
        Either False (default) or True denoting to use
        weights array slot in the h5 file specified by file_path
    seed : int, optional
        Default is 436. Sets the random seed for sampling.
    mode : {'train', 'validate', 'test'}, optional
        Default is `'train'`. The mode to run the sampler in.
    save_datasets : list(str), optional
        Default is `[]` the empty list. The list of modes for which we should
        save the sampled data to file (e.g. `["test", "validate"]`).
    output_dir : str or None, optional
        Default is None. The path to the directory where we should
        save sampled examples for a mode. If `save_datasets` is
        a non-empty list, `output_dir` must be specified. If
        the path in `output_dir` does not exist it will be created
        automatically.

    Attributes
    ----------
    target : selene_sdk.targets.Target
        The `selene_sdk.targets.Target` object holding the features that we
        would like to predict.
    sequence_length : int
        The length of the sequences to  train the model on.
    modes : list(str)
        The list of modes that the sampler can be run in.
    mode : str
        The current mode that the sampler is running in. Must be one of
        the modes listed in `modes`.

    Raises
    ------
    ValueError
            If `mode` is not a valid mode.
    """
    def __init__(self,
                 file_path,
                 features,
                 weights=False,
                 seed=436,
                 sequence_length=1001,
                 mode="train",
                 save_datasets=[],
                 output_dir=None):

        """
        Creates a new `OnlineSampler` object.
        """
        super(OnlineH5Sampler, self).__init__(
            features,
            save_datasets=save_datasets,
            output_dir=output_dir)

        self.weights = weights
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed + 1)

        # specifying a test holdout partition is optional
        self.modes.append("test")
        self._holdout_type = "user"

        if mode not in self.modes:
            raise ValueError(
                "Mode must be one of {0}. Input was '{1}'.".format(
                    self.modes, mode))
        self.mode = mode
        self.sequence_length = sequence_length

        self.n_features = len(self._features)

        self.target = H5Features(self._features)

        self._save_filehandles = {}

    def get_feature_from_index(self, index):
        """
        Returns the feature corresponding to an index in the feature
        vector.

        Parameters
        ----------
        index : int
            The index of the feature to retrieve the name for.

        Returns
        -------
        str
            The name of the feature occurring at the specified index.
        """
        return self.target.index_feature_dict[index]

    def get_sequence_from_encoding(self, encoding):
        """
        Gets the string sequence from the one-hot encoding
        of the sequence.

        Parameters
        ----------
        encoding : numpy.ndarray
            An :math:`L \\times N` array (where :math:`L` is the length
            of the sequence and :math:`N` is the size of the sequence
            type's alphabet) containing the one-hot encoding of the
            sequence.

        Returns
        -------
        str
            The sequence of :math:`L` characters decoded from the input.
        """
        BASES_ARR = np.array(['A', 'C', 'G', 'T'])
        UNK_BASE = "N"
        return sequence.encoding_to_sequence(encoding, BASES_ARR, UNK_BASE)

    def save_dataset_to_file(self, mode, close_filehandle=False):
        """
        Save samples for each partition (i.e. train/validate/test) to
        disk.

        Parameters
        ----------
        mode : str
            Must be one of the modes specified in `save_datasets` during
            sampler initialization.
        close_filehandle : bool, optional
            Default is False. `close_filehandle=True` assumes that all
            data corresponding to the input `mode` has been saved to
            file and `save_dataset_to_file` will not be called with
            `mode` again.
        """
        if mode not in self._save_datasets:
            return
        samples = self._save_datasets[mode]
        if mode not in self._save_filehandles:
            if self.mode == 'test':
                self._save_filehandles[mode] = open(
                    os.path.join(self._output_dir,
                                 "{0}_data.fa".format(mode)),
                    'w+')
            else:
                self._save_filehandles[mode] = open(
                    os.path.join(self._output_dir,
                                 "{0}_data.txt".format(mode)),
                    'w+')
        file_handle = self._save_filehandles[mode]
        while len(samples) > 0:
            cols = samples.pop(0)
            if self.mode == 'test':
                line = (">" + cols[1] + "\n" + cols[3])
            else:
                line = '\t'.join([str(c) for c in cols])
            file_handle.write("{0}\n".format(line))
        if close_filehandle:
            file_handle.close()

    def get_data_and_targets(self, batch_size, n_samples=None, mode=None):
        """
        This method fetches a subset of the data from the sampler,
        divided into batches. This method also allows the user to
        specify what operating mode to run the sampler in when fetching
        the data.

        Parameters
        ----------
        batch_size : int
            The size of the batches to divide the data into.
        n_samples : int or None, optional
            Default is None. The total number of samples to retrieve.
            If `n_samples` is None and the mode is `validate`, will
            set `n_samples` to 32000; if the mode is `test`, will set
            `n_samples` to 640000 if it is None. If the mode is `train`
            you must have specified a value for `n_samples`.
        mode : str, optional
            Default is None. The mode to run the sampler in when
            fetching the samples. See
            `selene_sdk.samplers.IntervalsSampler.modes` for more
            information. If None, will use the current mode `self.mode`.

        Returns
        -------
        sequences_and_targets, targets_matrix : \
        tuple(list(tuple(numpy.ndarray, numpy.ndarray)), numpy.ndarray)
            Tuple containing the list of sequence-target pairs, as well
            as a single matrix with all targets in the same order.
            Note that `sequences_and_targets`'s sequence elements are of
            the shape :math:`B \\times L \\times N` and its target
            elements are of the shape :math:`B \\times F`, where
            :math:`B` is `batch_size`, :math:`L` is the sequence length,
            :math:`N` is the size of the sequence type's alphabet, and
            :math:`F` is the number of features. Further,
            `target_matrix` is of the shape :math:`S \\times F`, where
            :math:`S =` `n_samples`.

        """
        if mode is not None:
            self.set_mode(mode)
        else:
            mode = self.mode
        sequences_and_targets = []
        if n_samples is None and mode == "validate":
            n_samples = 32000
        elif n_samples is None and mode == "test":
            n_samples = 640000

        n_batches = int(n_samples / batch_size)
        for _ in range(n_batches):
            inputs, targets, weights = self.sample(batch_size)
            sequences_and_targets.append((inputs, targets, weights))
        targets_mat = np.vstack([t for (s, t, u) in sequences_and_targets])
        weights_mat = np.vstack([u for (s, t, u) in sequences_and_targets])
        if mode in self._save_datasets:
            self.save_dataset_to_file(mode, close_filehandle=True)
        return sequences_and_targets, targets_mat, weights_mat

    def get_dataset_in_batches(self, mode, batch_size, n_samples=None):
        """
        This method returns a subset of the data for a specified run
        mode, divided into mini-batches.

        Parameters
        ----------
        mode : {'test', 'validate'}
            The mode to run the sampler in when fetching the samples.
            See `selene_sdk.samplers.IntervalsSampler.modes` for more
            information.
        batch_size : int
            The size of the batches to divide the data into.
        n_samples : int or None, optional
            Default is `None`. The total number of samples to retrieve.
            If `None`, it will retrieve 32000 samples if `mode` is validate
            or 640000 samples if `mode` is test or train.

        Returns
        -------
        sequences_and_targets, targets_matrix : \
        tuple(list(tuple(numpy.ndarray, numpy.ndarray)), numpy.ndarray)
            Tuple containing the list of sequence-target pairs, as well
            as a single matrix with all targets in the same order.
            The list is length :math:`S`, where :math:`S =` `n_samples`.
            Note that `sequences_and_targets`'s sequence elements are of
            the shape :math:`B \\times L \\times N` and its target
            elements are of the shape :math:`B \\times F`, where
            :math:`B` is `batch_size`, :math:`L` is the sequence length,
            :math:`N` is the size of the sequence type's alphabet, and
            :math:`F` is the number of features. Further,
            `target_matrix` is of the shape :math:`S \\times F`

        """
        return self.get_data_and_targets(
            batch_size, n_samples=n_samples, mode=mode)

    def get_validation_set(self, batch_size, n_samples=None):
        """
        This method returns a subset of validation data from the
        sampler, divided into batches.

        Parameters
        ----------
        batch_size : int
            The size of the batches to divide the data into.
        n_samples : int or None, optional
            Default is `None`. The total number of validation examples
            to retrieve. If `None`, 32000 examples are retrieved.

        Returns
        -------
        sequences_and_targets, targets_matrix : \
        tuple(list(tuple(numpy.ndarray, numpy.ndarray)), numpy.ndarray)
            Tuple containing the list of sequence-target pairs, as well
            as a single matrix with all targets in the same order.
            Note that `sequences_and_targets`'s sequence elements are of
            the shape :math:`B \\times L \\times N` and its target
            elements are of the shape :math:`B \\times F`, where
            :math:`B` is `batch_size`, :math:`L` is the sequence length,
            :math:`N` is the size of the sequence type's alphabet, and
            :math:`F` is the number of features. Further,
            `target_matrix` is of the shape :math:`S \\times F`, where
            :math:`S =` `n_samples`.

        """
        return self.get_dataset_in_batches(
            "validate", batch_size, n_samples=n_samples)

    def get_test_set(self, batch_size, n_samples=None):
        """
        This method returns a subset of testing data from the
        sampler, divided into batches.

        Parameters
        ----------
        batch_size : int
            The size of the batches to divide the data into.
        n_samples : int or None, optional
            Default is `None`. The total number of validation examples
            to retrieve. If `None`, 640000 examples are retrieved.

        Returns
        -------
        sequences_and_targets, targets_matrix : \
        tuple(list(tuple(numpy.ndarray, numpy.ndarray)), numpy.ndarray)
            Tuple containing the list of sequence-target pairs, as well
            as a single matrix with all targets in the same order.
            Note that `sequences_and_targets`'s sequence elements are of
            the shape :math:`B \\times L \\times N` and its target
            elements are of the shape :math:`B \\times F`, where
            :math:`B` is `batch_size`, :math:`L` is the sequence length,
            :math:`N` is the size of the sequence type's alphabet, and
            :math:`F` is the number of features. Further,
            `target_matrix` is of the shape :math:`S \\times F`, where
            :math:`S =` `n_samples`.


        Raises
        ------
        ValueError
            If no test partition of the data was specified during
            sampler initialization.
        """
        if "test" not in self.modes:
            raise ValueError("No test partition of the data was specified "
                             "during initialization. Cannot use method "
                             "`get_test_set`.")
        return self.get_dataset_in_batches("test", batch_size, n_samples)
