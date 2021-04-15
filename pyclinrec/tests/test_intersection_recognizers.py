import unittest

from unittest import TestCase

from pyclinrec.dictionary import StringDictionaryLoader
from pyclinrec.recognizer import InterDoubleMetaphoneConceptRecognizer, IntersStemConceptRecognizer


class TestRecognizers(TestCase):

    def setUp(self):
        self.dictionary_loader = StringDictionaryLoader([("1", "bright cat"), ("2", "Brighton"), ("3", "mechanical")])

    def generic_english_tests(self, recognizer):
        # Base annotation
        spans, tokens, annotations = recognizer.recognize("The bright cat is from Brighton.")

        annotations = list(annotations)
        annotations.sort(key=lambda a: a.start)

        self.assertTrue(spans[1][0] == annotations[0].start and spans[2][1] == annotations[0].end and spans[-2][0] == \
                        annotations[1].start and spans[-2][1] == annotations[1].end)

        # Leading space
        spans, tokens, annotations = recognizer.recognize(" mechanical  Brighton")
        annotations = list(annotations)

    def test_doublemetaphone_recognize_english(self):
        recognizer = InterDoubleMetaphoneConceptRecognizer(self.dictionary_loader, "pyclinrec/stopwordsen.txt",
                                                           "pyclinrec/termination_termsen.txt")
        recognizer.initialize()
        self.generic_english_tests(recognizer)

    def test_stem_recognize(self):
        recognizer = IntersStemConceptRecognizer(self.dictionary_loader, "pyclinrec/stopwordsen.txt",
                                                 "pyclinrec/termination_termsen.txt")
        recognizer.initialize()

        self.generic_english_tests(recognizer)


if __name__ == '__main__':
    unittest.main()
