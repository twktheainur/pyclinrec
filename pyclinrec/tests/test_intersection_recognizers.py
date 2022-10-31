import unittest

from unittest import TestCase

from pyclinrec.dictionary import StringDictionaryLoader
from pyclinrec.recognizer import IntersStemConceptRecognizer

import logging

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
            [("1", "bright cat"), ("2", "Brighton"), ("3", "mechanical"), ('4', "cet"), ('5', "caets"), ('6', 'cat'),
             ('7', 'a')])

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
        spans, tokens, annotations = recognizer.annotate("A bright cat loves Brighton's bright lights.")
        annotations = list(annotations)
        annotations.sort(key=lambda a: a.matched_length)
        self.assertTrue(len(annotations[0].matched_text) > 3)
        logging.info(f"Test successful for {recognizer.__class__} INFO")
        logging.debug(f"Test successful for {recognizer.__class__} DEBUG")

    def test_stem_recognize_english_1(self):
        recognizer = IntersStemConceptRecognizer(self.dictionary_loader, "pyclinrec/stopwordsen.txt",
                                                 "pyclinrec/termination_termsen.txt", filters=[])
        recognizer.initialize()

        self._generic_english_1(recognizer)

    def test_stem_recognize_english_leading_space_double_spaces(self):
        recognizer = IntersStemConceptRecognizer(self.dictionary_loader, "pyclinrec/stopwordsen.txt",
                                                 "pyclinrec/termination_termsen.txt", filters=[])
        recognizer.initialize()

        self._generic_english_leading_space_double_spaces(recognizer)

    def test_agrovoc_sentence_end_bug(self):
        dictionary_loader = StringDictionaryLoader(
            [("1", "spectroscopy"), ("2", "technique"), ("3", "according")])
        recognizer = IntersStemConceptRecognizer(dictionary_loader, "pyclinrec/stopwordsen.txt",
                                                 "pyclinrec/termination_termsen.txt", filters=[])
        recognizer.initialize()
        text = "infrared spectroscopy technique (NIRS). According to"
        spans, tokens, annotations = recognizer.annotate(text)
        annotations = list(annotations)
        annotations.sort(key=lambda a: a.start)
        self.assertTrue (spans[-2][0] == annotations[-1].start and spans[-2][1] == annotations[-1].end and annotations[
            -1].matched_text == "According")

    def test_stem_recognize_english_short_match_filter(self):
        recognizer = IntersStemConceptRecognizer(self.dictionary_loader, "pyclinrec/stopwordsen.txt",
                                                 "pyclinrec/termination_termsen.txt")
        recognizer.initialize()

        self._generic_english_single_stem_artifacts(recognizer)

    def test_double_word_match_bug(self):
        text = "Secondly, plant-plant interactions influence allelochemical fun"
        dictionary_loader = StringDictionaryLoader(
            [("1", "blasts (of plants)"), ("2", "emergent aquatic plants"), ("3", "plant structure"),
             ('4', "submerged aquatic plants"), ('5', 'allelochemical'), ('6', 'plant-plant interaction')])
        recognizer = IntersStemConceptRecognizer(dictionary_loader, "pyclinrec/stopwordsen.txt",
                                                 "pyclinrec/termination_termsen.txt")
        recognizer.initialize()
        spans, tokens, annotations = recognizer.annotate(text)
        self.assertTrue(len(annotations) == 2)

    # def test_approxtrie_recognize_english_1(self):
    #     recognizer = TrieApproxRecognizer(self.dictionary_loader)
    #     recognizer.initialize()
    #
    #     self._generic_english_1(recognizer)


if __name__ == '__main__':
    unittest.main()
