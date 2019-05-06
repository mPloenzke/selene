"""
This module provides the `fastaSampler` class and supporting
methods.
"""
from collections import namedtuple
import logging
import random
import os

import pyfaidx
import numpy as np

from .online_fasta_sampler import OnlineFastaSampler
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


class FastaSampler(OnlineFastaSampler):
    """
    Draws samples from pre-specified fasta sequences.

    Parameters
    ----------
    target_path : str
        Path to fasta (`*.fa`) of genomic sequences.
    features : list(str)
        List of distinct features that we aim to predict.
    sample_negative : bool, optional
        Default is `False`. This tells the sampler whether negative
        examples (i.e. with no positive labels) should be drawn when
        generating samples. If `True`, both negative and positive
        samples will be drawn. If `False`, only samples with at least
        one positive label will be drawn.
    seed : int, optional
        Default is 436. Sets the random seed for sampling.
    validation_holdout : float, optional
        Default is `0.10`. Holdout can be is
        proportional. Specify a percentage
        between (0.0, 1.0). Typically 0.10 or 0.20.
    test_holdout : float, optional
        Default is `0.10`. See documentation for
        `validation_holdout` for additional information.
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
    sample_from_intervals : list(tuple(str, int, int))
        A list of coordinates that specify the intervals we can draw
        samples from.
    sample_negative : bool
        Whether negative examples (i.e. with no positive label) should
        be drawn when generating samples. If `True`, both negative and
        positive samples will be drawn. If `False`, only samples with at
        least one positive label will be drawn.
    validation_holdout : float
        The samples to hold out for validating model performance. 
        This is the fraction of total samples
        that will be held out.
    test_holdout : loat
        The samples to hold out for testing model performance. See the
        documentation for `validation_holdout` for more details.
    sequence_length : int
        The length of the sequences to  train the model on.
    modes : list(str)
        The list of modes that the sampler can be run in.
    mode : str
        The current mode that the sampler is running in. Must be one of
        the modes listed in `modes`.

    """
    def __init__(self,
                 target_path,
                 features,
                 sample_negative=False,
                 seed=436,
                 validation_holdout=0.10,
                 test_holdout=0.10,
                 sequence_length=1000,
                 mode="train",
                 save_datasets=["test"],
                 output_dir=None):
        """
        Constructs a new `FastaSampler` object.
        """
        super(FastaSampler, self).__init__(
            target_path,
            features,
            seed=seed,
            validation_holdout=validation_holdout,
            test_holdout=test_holdout,
            sequence_length=sequence_length,
            mode=mode,
            save_datasets=save_datasets,
            output_dir=output_dir)

        self._sample_from_mode = {}
        self._randcache = {}
        for mode in self.modes:
            self._sample_from_mode[mode] = None
            self._randcache[mode] = {"cache_indices": None, "sample_next": 0}

        self.sample_from_fastas = []
        self.fasta_lengths = []

        self._partition_dataset_proportion(target_path)

        for mode in self.modes:
            self._update_randcache(mode=mode)

        self.sample_negative = sample_negative

    def _partition_dataset_proportion(self, fastas_path):
        """
        When holdout sets are created by randomly sampling a proportion
        of the data, this method is used to divide the data into
        train/test/validate subsets.

        Parameters
        ----------
        fastas_path : str
            The path to the file that contains the fasta sequences to sample
            from. In this file, each fasta format should be followed.

        """
        fasta_files = os.listdir(fastas_path)
        for i, fasta_in in enumerate(fasta_files):
            if not '.fa' in fasta_in:
                continue
            if '.fai' in fasta_in:
                continue
            fasta_file = pyfaidx.Fasta(os.path.join(fastas_path,fasta_in), duplicate_action="first")
            for j, fasta_record in enumerate(fasta_file): 
                cur_sequence = str.upper(str(fasta_record))
                cur_name = fasta_record.name
                cur_label = cur_name.split('-')[0]
                self.sample_from_fastas.append((fasta_in, cur_name, cur_label))
                self.fasta_lengths.append(len(cur_sequence))
        n_sequences = len(self.sample_from_fastas)

        # all indices in the fasta list are shuffled
        select_indices = list(range(n_sequences))
        np.random.shuffle(select_indices)

        # the first section of indices is used as the validation set
        n_indices_validate = int(n_sequences * self.validation_holdout)
        val_indices, val_weights = get_indices_and_probabilities(
            self.fasta_lengths, select_indices[:n_indices_validate])
        self._sample_from_mode["validate"] = SampleIndices(
            val_indices, val_weights)

        if self.test_holdout is not None:
            # if applicable, the second section of indices is used as the
            # test set
            n_indices_test = int(n_sequences * self.test_holdout)
            test_indices_end = n_indices_test + n_indices_validate
            test_indices, test_weights = get_indices_and_probabilities(
                self.fasta_lengths,
                select_indices[n_indices_validate:test_indices_end])
            self._sample_from_mode["test"] = SampleIndices(
                test_indices, test_weights)

            # remaining indices are for the training set
            tr_indices, tr_weights = get_indices_and_probabilities(
                self.fasta_lengths, select_indices[test_indices_end:])
            self._sample_from_mode["train"] = SampleIndices(
                tr_indices, tr_weights)
        else:
            # remaining indices are for the training set
            tr_indices, tr_weights = get_indices_and_probabilities(
                self.fasta_lengths, select_indices)
            self._sample_from_mode["train"] = SampleIndices(
                tr_indices, tr_weights)

    def _retrieve(self, file, sequence_name, label_vec):
        """
        Retrieves samples from the fast file.

        Parameters
        ----------
        sequence_name : int
            The row name to query. 
        file : int
            The file to query.

        Returns
        -------
        retrieved_seq, retrieved_targets : \
        tuple(numpy.ndarray, list(numpy.ndarray))
            A tuple containing the numeric representation of the
            sequence and labels.

        """
        BASES_ARR = np.array(['A', 'C', 'G', 'T'])
        UNK_BASE = "N"
        BASE_TO_INDEX = {
            'A': 0, 'C': 1, 'G': 2, 'T': 3,
            'a': 0, 'c': 1, 'g': 2, 't': 3,
        }

        file_num = self.target.file_index_dict[file]
        nucleotide_sequence = str(self.target.data[file_num][sequence_name])
        retrieved_targets = self.target.get_feature_data(label_vec)
        
        if not self.sample_negative and np.sum(retrieved_targets) == 0:
            logger.info("Negative sequence found. Sampling again.")
            return None

        retrieved_seq = sequence.sequence_to_encoding(nucleotide_sequence, BASE_TO_INDEX, BASES_ARR)
        if retrieved_seq.shape[0] == 0:
            logger.info("Sequence could not be retrieved. Sampling again.")
            return None
        elif np.sum(retrieved_seq) / float(retrieved_seq.shape[0]) < 0.60:
            logger.info("Over 30% of the bases are ambiguous ('N'). "
                        "Sampling again.")
            return None

        if self.mode in self._save_datasets:
            if self.mode == 'test':
                self._save_datasets[self.mode].append(
                    [file, sequence_name, label_vec, nucleotide_sequence])
            else:
                self._save_datasets[self.mode].append(
                    [file, sequence_name, label_vec])
            if len(self._save_datasets[self.mode]) > 200000:
                self.save_dataset_to_file(self.mode)
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

            rand_fasta_index = \
                self._randcache[self.mode]["cache_indices"][sample_index]
            self._randcache[self.mode]["sample_next"] += 1

            fasta_info = self.sample_from_fastas[rand_fasta_index]
            fasta_length = self.fasta_lengths[rand_fasta_index]

            fasta_file = fasta_info[0]
            seq_name = fasta_info[1]
            labels = fasta_info[2]

            retrieve_output = self._retrieve(fasta_file, seq_name, labels)
            if not retrieve_output:
                continue
            seq, seq_targets = retrieve_output
            sequences[n_samples_drawn, :, :] = seq
            targets[n_samples_drawn, :] = seq_targets
            n_samples_drawn += 1
        return (sequences, targets)
