import unittest

from pyclinrec.dictionary import StringDictionaryLoader
from pyclinrec.recognizer import InterDoubleMetaphoneConceptRecognizer, IntersStemConceptRecognizer
from unittest import TestCase


class TestRecognizers(TestCase):

    def setUp(self):
        self.dictionary_loader = StringDictionaryLoader([("1", "bright cat"), ("2", "Brighton")])

    def test_doublemetaphone_recognize(self):
        recognizer = InterDoubleMetaphoneConceptRecognizer(self.dictionary_loader, "../stopwordsen.txt",
                                                           "../termination_termsen.txt")
        recognizer.initialize()

        spans, tokens, annotations = recognizer.recognize("The bright cat is from Brighton.")

        annotations = list(annotations)
        annotations.sort(key=lambda a: a.start)

        self.assertTrue(spans[1][0] == annotations[0].start and spans[2][1] == annotations[0].end and spans[-2][0] == \
                        annotations[1].start and spans[-2][1] == annotations[1].end)

    def test_stem_recognize(self):
        recognizer = IntersStemConceptRecognizer(self.dictionary_loader, "../stopwordsen.txt",
                                                 "../termination_termsen.txt")
        recognizer.initialize()

        spans, tokens, annotations = recognizer.recognize("The bright cat is from Brighton.")

        annotations = list(annotations)
        annotations.sort(key=lambda a: a.start)

        self.assertTrue(spans[1][0] == annotations[0].start and spans[2][1] == annotations[0].end and spans[-2][0] == \
                        annotations[1].start and spans[-2][1] == annotations[1].end)


if __name__ == '__main__':
    unittest.main()
