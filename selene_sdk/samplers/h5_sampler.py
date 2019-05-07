"""
This module provides the `H5Sampler` class and supporting
methods.
"""
from collections import namedtuple
import logging
import random
import os

import numpy as np
import h5py

from .online_h5_sampler import OnlineH5Sampler
from ..sequences import sequence
from ..utils import get_indices_and_probabilities

logger = logging.getLogger(__name__)


SampleIndices = namedtuple(
    "SampleIndices", ["indices", "weights"])
"""
A tuple containing the indices for some samples, and a weight to
allot to each index when randomly drawing from them.

Parameters
----------
indices : list(int)
    The numeric index of each sample.
weights : list(float)
    The amount of weight assigned to each sample.

Attributes
----------
indices : list(int)
    The numeric index of each sample.
weights : list(float)
    The amount of weight assigned to each sample.

"""


class H5Sampler(OnlineH5Sampler):
    """
    Draws samples from pre-specified h5 one-hot encoded arrays.

    Parameters
    ----------
    file_path : str
        Path to h5 (`*.h5`) file.
    features : list(str)
        List of distinct features that we aim to predict.
    seed : int, optional
        Default is 436. Sets the random seed for sampling.
    mode : {'train', 'validate', 'test'}
        Default is `'train'`. The mode to run the sampler in.
    save_datasets : list of str
        Default is `["test"]`. The list of modes for which we should
        save the sampled data to file.
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

    """
    def __init__(self,
                 file_path,
                 features,
                 seed=436,
                 sequence_length=1000,
                 mode="train",
                 save_datasets=["test"],
                 output_dir=None):
        """
        Constructs a new `FastaSampler` object.
        """
        super(H5Sampler, self).__init__(
            file_path,
            features,
            seed=seed,
            sequence_length=sequence_length,
            mode=mode,
            save_datasets=save_datasets,
            output_dir=output_dir)

        self._sample_from_mode = {}
        self._randcache = {}
        for mode in self.modes:
            self._sample_from_mode[mode] = None
            self._randcache[mode] = {"cache_indices": None, "sample_next": 0}

        self._load_h5_dataset(file_path)

        for mode in self.modes:
            self._update_randcache(mode=mode)

    def _load_h5_dataset(self, h5_path):
        """
        Holdout sets are created by extracting the saved datasets
        from the h5 file, this method is then used to store the data into
        train/test/validate subsets.

        Parameters
        ----------
        h5_path : str
            The path to the h5 file that contains the one-hot sequences to sample
            from.

        """
        trainmat = h5py.File(h5_path, 'r')
        X_train = np.array(trainmat['X_train']).astype(np.float32)
        y_train = np.array(trainmat['Y_train']).astype(np.float32)
        X_valid = np.array(trainmat['X_valid']).astype(np.float32)
        y_valid = np.array(trainmat['Y_valid']).astype(np.int32)
        X_test = np.array(trainmat['X_test']).astype(np.float32)
        y_test = np.array(trainmat['Y_test']).astype(np.int32)

        if (len(X_train.shape)==3):
            X_train = X_train.transpose([0,2,1])
            X_valid = X_valid.transpose([0,2,1])
            X_test = X_test.transpose([0,2,1])
        elif (len(X_train.shape)==4):
            X_train = X_train.transpose([0,2,3,1])
            X_valid = X_valid.transpose([0,2,3,1])
            X_test = X_test.transpose([0,2,3,1])
        
        X_all = np.vstack((X_train, X_valid, X_test))
        y_all = np.vstack((y_train, y_valid, y_test))

        self.data = X_all
        self.target_mat = y_all

        n_sequences = X_all.shape[0]
        select_indices = list(range(n_sequences))

        # the first section of indices is used as the validation set
        n_indices_validate = X_valid.shape[0]
        val_indices = range(X_train.shape[0],X_train.shape[0]+n_indices_validate)
        val_weights = np.ones(n_indices_validate)/n_indices_validate
        self._sample_from_mode["validate"] = SampleIndices(
            val_indices, val_weights)

        # if applicable, the second section of indices is used as the
        # test set
        n_indices_test = X_test.shape[0]
        test_indices = range(X_train.shape[0]+n_indices_validate,X_train.shape[0]+n_indices_validate+n_indices_test)
        test_weights = np.ones(n_indices_test)/n_indices_test
        self._sample_from_mode["test"] = SampleIndices(
            test_indices, test_weights)

        # remaining indices are for the training set
        tr_indices = range(0,X_train.shape[0])
        tr_weights = np.ones(X_train.shape[0])/X_train.shape[0]
        self._sample_from_mode["train"] = SampleIndices(
            tr_indices, tr_weights)

    def _retrieve(self, rand_seq_index):
        """
        Retrieves samples from the data array.

        Parameters
        ----------
        rand_seq_index : int
            The row number 

        Returns
        -------
        retrieved_seq, retrieved_targets : \
        tuple(numpy.ndarray, list(numpy.ndarray))
            A tuple containing the numeric representation of the
            sequence as well as labels.

        """

        retrieved_seq = self.data[rand_seq_index,:,:]
        retrieved_targets = self.target_mat[rand_seq_index,:]

        if self.mode in self._save_datasets:
            if self.mode == 'test':
                seq_name = np.array(self._features)[retrieved_targets>0]
                labs = []
                for ll in seq_name: labs.append(ll.split('|')[1])
                seq_name = '_'.join(labs) + "-" +str(random.randint(1,1e6))
                if (len(retrieved_seq.shape)==3):
                    self._save_datasets[self.mode].append(
                        ['test', seq_name, retrieved_targets, self.get_sequence_from_encoding(retrieved_seq[:,0,:])])
                elif (len(retrieved_seq.shape)==2):
                    self._save_datasets[self.mode].append(
                        ['test', seq_name, retrieved_targets, self.get_sequence_from_encoding(retrieved_seq[:,:])])

        return (retrieved_seq, retrieved_targets)

    def _update_randcache(self, mode=None):
        """
        Updates the cache of indices of intervals. This allows us
        to randomly sample from our data without having to use a
        fixed-point approach or keeping all labels in memory.

        Parameters
        ----------
        mode : str or None, optional
            Default is `None`. The mode that these samples should be
            used for. See `selene_sdk.samplers.IntervalsSampler.modes` for
            more information.

        """
        if not mode:
            mode = self.mode
        self._randcache[mode]["cache_indices"] = np.random.choice(
            self._sample_from_mode[mode].indices,
            size=len(self._sample_from_mode[mode].indices),
            replace=True,
            p=self._sample_from_mode[mode].weights)
        self._randcache[mode]["sample_next"] = 0

    def sample(self, batch_size=1):
        """
        Randomly draws a mini-batch of examples and their corresponding
        labels.

        Parameters
        ----------
        batch_size : int, optional
            Default is 1. The number of examples to include in the
            mini-batch.

        Returns
        -------
        sequences, targets : tuple(numpy.ndarray, numpy.ndarray)
            A tuple containing the numeric representation of the
            sequence examples and their corresponding labels. The
            shape of `sequences` will be
            :math:`B \\times L \\times N`, where :math:`B` is
            `batch_size`, :math:`L` is the sequence length, and
            :math:`N` is the size of the sequence type's alphabet.
            The shape of `targets` will be :math:`B \\times F`,
            where :math:`F` is the number of features.

        """
        sequences = np.zeros((batch_size, self.sequence_length, 4))
        targets = np.zeros((batch_size, self.n_features))
        n_samples_drawn = 0
        while n_samples_drawn < batch_size:
            sample_index = self._randcache[self.mode]["sample_next"]
            if sample_index == len(self._sample_from_mode[self.mode].indices):
                self._update_randcache()
                sample_index = 0

            rand_seq_index = \
                self._randcache[self.mode]["cache_indices"][sample_index]
            self._randcache[self.mode]["sample_next"] += 1

            retrieve_output = self._retrieve(rand_seq_index)
            if not retrieve_output:
                continue
            seq, seq_targets = retrieve_output
            if len(seq.shape) == 3:
                sequences[n_samples_drawn, :, :] = seq[:,0,:]
            elif len(seq.shape) == 2: 
                sequences[n_samples_drawn, :, :] = seq
            targets[n_samples_drawn, :] = seq_targets
            n_samples_drawn += 1
        return (sequences, targets)
