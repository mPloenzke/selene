import os
import unittest

from data_utils import GenomicFeatures
from data_utils.genomic_features import _any_positive_rows, \
    _is_positive_row, _get_feature_data


class TestGenomicFeatures(unittest.TestCase):

    def setUp(self):
        self.features = [
            "CTCF", "eGFP-FOS", "GABP", "Pbx3", "Pol2", "TBP"
        ]
        self.feature_index_map = {
            "CTCF": 0, "eGFP-FOS": 1, "GABP": 2, "Pbx3": 3, "Pol2": 4, "TBP": 5
        }

        # CTCF only, between 16110 and 16239
        self.rows_example1 =  \
            [("chr1", "16110", "16190", ".", "CTCF", "662"),  # len 70
             ("chr1", "16128", "16158", ".", "CTCF", "631"),  # len 30
             ("chr1", "16149", "16239", ".", "CTCF", "628")]  # len 90

        # CTCF only, between 91128 and 91358
        self.rows_example2 =  \
            [("chr2", "91128", "91358", ".", "CTCF", "631"),  # len 200
             ("chr2", "91130", "91239", ".", "CTCF", "628"),  # len 109
             ("chr2", "91156", "91310", ".", "CTCF", "662")]  # len 154

        # multiple features, between 8533 and 9049
        self.rows_example3 = \
            [("chr3", "8533", "8817", ".", "eGFP-FOS", "590"),  # len 284
             ("chr3", "8541", "8651", ".", "GABP", "220"),      # len 110
             ("chr3", "8574", "8629", ".", "Pol2", "229"),      # len 145
             ("chr3", "8619", "9049", ".", "CTCF", "44"),       # len 430
             ("chr3", "8620", "8680", ".", "TBP", "545"),       # len 60
             ("chr3", "8645", "8720", ".", "TBP", "546")]       # len 75

    def get_feature_rows(self, chrom, start, end):
        """This function disregards (`start`, `end`) input
        """
        if chrom is None:
            return None

        if chrom == "chr1":
            return self.rows_example1
        elif chrom == "chr2":
            return self.rows_example2
        elif chrom == "chr3":
            return self.rows_example3
        else:
            return []

    ############################################
    # Correctness tests for `_is_positive_row`
    ############################################

    def test__is_positive_row_false(self):
        query_start, query_end = (16150, 16351)  # len 201
        feat_start, feat_end = (16110, 16190)    # len 80
        threshold = 0.50
        self.assertFalse(
            _is_positive_row(
                query_start, query_end, feat_start, feat_end, threshold))

    def test__is_positive_row_true_eq_threshold(self):
        query_start, query_end = (16110, 16309)  # len 199
        feat_start, feat_end = (16110, 16190)    # len 80
        threshold = 0.40
        self.assertTrue(
            _is_positive_row(
                query_start, query_end, feat_start, feat_end, threshold))

    def test__is_positive_row_true_gt_threshold(self):
        query_start, query_end = (16110, 16311)  # len 201
        feat_start, feat_end = (16110, 16290)    # len 170
        threshold = 0.80
        self.assertTrue(
            _is_positive_row(
                query_start, query_end, feat_start, feat_end, threshold))

    ############################################
    # Correctness tests for `_any_positive_rows`
    ############################################

    def test__any_positive_rows_none_rows(self):
        rows = None
        query_start, query_end = (10, 100)
        threshold = 0.50
        self.assertFalse(
            _any_positive_rows(rows, query_start, query_end, threshold))

    def test__any_positive_rows_empty_rows(self):
        rows = []
        query_start, query_end = (10, 100)
        threshold = 0.50
        self.assertFalse(
            _any_positive_rows(rows, query_start, query_end, threshold))

    def test__any_positive_rows_false(self):
        rows = self.rows_example1
        query_start, query_end = (16150, 16351)
        threshold = 0.50
        self.assertFalse(
            _any_positive_rows(rows, query_start, query_end, threshold))

    def test__any_positive_rows_true(self):
        rows = self.rows_example1
        query_start, query_end = (16150, 16351)
        threshold = 0.40
        self.assertTrue(
            _any_positive_rows(rows, query_start, query_end, threshold))

    ############################################
    # Correctness tests for `_get_feature_data`
    ############################################

    def test__get_feature_data_none_rows(self):
        query_chrom, query_start, query_end = (None, 10, 211)
        threshold = 0.50

        expected_encoding = [0, 0, 0, 0, 0, 0]
        observed_encoding = _get_feature_data(
            query_chrom, query_start, query_end, threshold,
            self.feature_index_map, self.get_feature_rows)

        self.assertSequenceEqual(
            observed_encoding.tolist(), expected_encoding)

    def test__get_feature_data_empty_rows(self):
        query_chrom, query_start, query_end = ("chr7", 10, 211)
        threshold = 0.50

        expected_encoding = [0, 0, 0, 0, 0, 0]
        observed_encoding = _get_feature_data(
            query_chrom, query_start, query_end, threshold,
            self.feature_index_map, self.get_feature_rows)

        self.assertSequenceEqual(
            observed_encoding.tolist(), expected_encoding)

    def test__get_feature_data_single_feat_positive(self):
        query_chrom, query_start, query_end = ("chr1", 16100, 16350)
        threshold = 0.50

        expected_encoding = [1, 0, 0, 0, 0, 0]
        observed_encoding = _get_feature_data(
            query_chrom, query_start, query_end, threshold,
            self.feature_index_map, self.get_feature_rows)

        self.assertSequenceEqual(
            observed_encoding.tolist(), expected_encoding)

    def test__get_feature_data_no_feat_positive(self):
        query_chrom, query_start, query_end = ("chr2", 91027, 91228)
        threshold = 0.50

        expected_encoding = [0, 0, 0, 0, 0, 0]
        observed_encoding = _get_feature_data(
            query_chrom, query_start, query_end, threshold,
            self.feature_index_map, self.get_feature_rows)

        self.assertSequenceEqual(
            observed_encoding.tolist(), expected_encoding)

    def test__get_feature_data_multiple_feats_positive(self):
        query_chrom, query_start, query_end = ("chr3", 8619, 8719)
        threshold = 0.50

        expected_encoding = [1, 1, 0, 0, 0, 1]
        observed_encoding = _get_feature_data(
            query_chrom, query_start, query_end, threshold,
            self.feature_index_map, self.get_feature_rows)

        self.assertSequenceEqual(
            observed_encoding.tolist(), expected_encoding)

    ############################################
    # Integration tests for `GenomicFeatures` class methods
    ############################################

    def test_GenomicFeatures_query_is_positive(self):
        # path assumes tests are run from the top-level Github dir.
        query_features = GenomicFeatures(
            os.path.join(".", "tests",
                         "files", "ChIP_CTCF_6feats",
                         "sorted_aggregate.bed.gz"),
            self.features)

        query_chrom, query_start, query_end = ("chr1", 29200, 29570)
        threshold = 0.50

        observed_is_positive = query_features.is_positive(
            query_chrom, query_start, query_end, threshold)

        self.assertTrue(observed_is_positive)

    def test_GenomicFeatures_query_get_feature_data(self):
        # path assumes tests are run from the top-level Github dir.
        query_features = GenomicFeatures(
            os.path.join(".", "tests",
                         "files", "ChIP_CTCF_6feats",
                         "sorted_aggregate.bed.gz"),
            self.features)

        query_chrom, query_start, query_end = ("chr10", 63348553, 63349171)
        threshold = 0.50
        expected_get_feature_data = [0, 0, 1, 0, 0, 0]
        observed_get_feature_data = query_features.get_feature_data(
            query_chrom, query_start, query_end, threshold)

        self.assertSequenceEqual(
            observed_get_feature_data.tolist(), expected_get_feature_data)


if __name__ == "__main__":
    unittest.main()