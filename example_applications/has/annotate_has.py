from pyclinrec.dictionary import MgrepDictionaryLoader

loader = MgrepDictionaryLoader("data/dictionary.tsv")

from pyclinrec.recognizer import IntersStemConceptRecognizer

recognizer = IntersStemConceptRecognizer(dictionary_loader=loader,
                                         stop_words_file="data/stopwordsfr.txt",
                                         termination_terms_file="data/termination_termsfr.txt")
recognizer.initialize()

with open("data/allText.txt") as f:
    annotations = recognizer.match_mentions(f.read())
    print(annotations)
