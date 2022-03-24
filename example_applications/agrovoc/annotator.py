from pyclinrec.dictionary import MgrepDictionaryLoader
from pyclinrec.recognizer import IntersStemConceptRecognizer


class AgrovocAnnotator:
    def __init__(self, dictionary_file, language="en"):
        dictionary_loader = MgrepDictionaryLoader(dictionary_file)

        self.concept_recognizer = IntersStemConceptRecognizer(dictionary_loader,
                                                              f"data/stopwords{language}.txt",
                                                              f"data/termination_terms{language}.txt")

        self.concept_recognizer.initialize()

    def annotate(self, text):
        return self.concept_recognizer.annotate(text)
