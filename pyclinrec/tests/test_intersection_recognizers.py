import unittest

from unittest import TestCase

from pyclinrec.dictionary import StringDictionaryLoader
from pyclinrec.recognizer import IntersStemConceptRecognizer

import logging

from string_matching_recognizers import TrieApproxRecognizer

logger = logging.getLogger()
fhandler = logging.FileHandler(filename='tests.log', mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.DEBUG)


class TestRecognizers(TestCase):

    def setUp(self):
        logging.info("Starting tests for pyclinrec.recognizers")
        self.dictionary_loader = StringDictionaryLoader(
            [("1", "bright cat"), ("2", "Brighton"), ("3", "mechanical"), ('4', "cet"), ('5', "caets"), ('6', 'cat'), ])

    def _generic_english_1(self, recognizer):
        # Base annotation
        spans, tokens, annotations = recognizer.annotate("The bright cat is from Brighton. Cats are like that!")

        annotations = list(annotations)
        annotations.sort(key=lambda a: a.start)

        self.assertTrue(spans[1][0] == annotations[0].start and spans[2][1] == annotations[0].end and
                        spans[2][0] == annotations[1].start and spans[2][1] == annotations[1].end and
                        spans[5][0] == annotations[2].start and spans[5][1] == annotations[2].end and
                        spans[7][0] == annotations[3].start and spans[7][1] == annotations[3].end)
        logging.info(f"Test successful for {recognizer.__class__} INFO")
        logging.debug(f"Test successful for {recognizer.__class__} DEBUG")
        # Leading space
        spans, tokens, annotations = recognizer.annotate(" mechanical  Brighton")
        annotations = list(annotations)
        annotations.sort(key=lambda a: a.start)
        self.assertTrue(spans[1][0] == annotations[0].start and spans[1][1] == annotations[0].end)

    def _generic_english_leading_space_double_spaces(self, recognizer):
        # Leading space
        spans, tokens, annotations = recognizer.annotate(" mechanical  Brighton")
        annotations = list(annotations)
        annotations.sort(key=lambda a: a.start)
        self.assertTrue(spans[1][0] == annotations[0].start and spans[1][1] == annotations[0].end and
                        spans[3][0] == annotations[1].start and spans[3][1] == annotations[1].end)
        logging.info(f"Test successful for {recognizer.__class__} INFO")
        logging.debug(f"Test successful for {recognizer.__class__} DEBUG")

    def _generic_english_single_stem_artifacts(self, recognizer):
        spans, tokens, annotations = recognizer.annotate("Bright children love Brighton's bright lights.")
        annotations = list(annotations)
        annotations.sort(key=lambda a: a.start)
        self.assertTrue(spans[3][0] == annotations[0].start and spans[3][1] == annotations[0].end)
        logging.info(f"Test successful for {recognizer.__class__} INFO")
        logging.debug(f"Test successful for {recognizer.__class__} DEBUG")

    def test_stem_recognize_english_1(self):
        recognizer = IntersStemConceptRecognizer(self.dictionary_loader, "pyclinrec/stopwordsen.txt",
                                                 "pyclinrec/termination_termsen.txt")
        recognizer.initialize()

        self._generic_english_1(recognizer)

    def test_stem_recognize_english_leading_space_double_spaces(self):
        recognizer = IntersStemConceptRecognizer(self.dictionary_loader, "pyclinrec/stopwordsen.txt",
                                                 "pyclinrec/termination_termsen.txt")
        recognizer.initialize()

        self._generic_english_leading_space_double_spaces(recognizer)

    def test_stem_recognize_english_single_stem_artifacts(self):
        recognizer = IntersStemConceptRecognizer(self.dictionary_loader, "pyclinrec/stopwordsen.txt",
                                                 "pyclinrec/termination_termsen.txt")
        recognizer.initialize()

        self._generic_english_single_stem_artifacts(recognizer)

    # def test_approxtrie_recognize_english_1(self):
    #     recognizer = TrieApproxRecognizer(self.dictionary_loader)
    #     recognizer.initialize()
    #
    #     self._generic_english_1(recognizer)


if __name__ == '__main__':
    unittest.main()
