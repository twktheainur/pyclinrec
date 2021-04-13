from dictionary import StringDictionaryLoader
from intersection_recognizers import InterDoubleMetaphoneConceptRecognizer


class TestRecognizers:
    def test_doublemetaphone_recognize(self):
        dictionary_loader = StringDictionaryLoader([("1", "bright cat"), ("2", "Brighton")])
        recognizer = InterDoubleMetaphoneConceptRecognizer(dictionary_loader, "../data/stopwordsen.txt",
                                                           "../data/termination_termsen.txt")
        recognizer.initialize()

        spans, tokens, annotations = recognizer.recognize("The bright cat is from Brighton.")

        annotations = list(annotations)
        annotations.sort(key=lambda a: a.start)

        assert spans[1][0] == annotations[0].start and spans[2][1] == annotations[0].end and spans[-2][0] == \
               annotations[1].start and spans[-2][1] == annotations[1].end
