from typing import List, Tuple, Set

import regex

from pyclinrec.dictionary import DictionaryLoader
from pyclinrec.recognizer import ConceptRecognizer, Annotation
from pyclinrec.spacy_utils import span_tokenize
from recognizer import AnnotationFilter
from lexpy.trie import Trie


class TrieApproxRecognizer(ConceptRecognizer):
    def __init__(self, dictionary_loader: DictionaryLoader,
                 language="en", filters: List[AnnotationFilter] = None):
        """
        Parameters
        ----------
            dictionary_loader: DictionaryLoader
                The dictionary loader that will provide the dictionary contents
            language: str
                The language of the text that will processed (affects the choice of tokenner and stemmer).
            filters: List[AnnotationFilter]
                A list of filters to apply post recognition
        """
        super().__init__(dictionary_loader, language=language, filters=filters)
        self.punctuation_remove = regex.compile(r'[\p{C}|\p{M}|\p{P}|\p{S}|\p{Z}]+', regex.UNICODE)
        self.label_concept_index = {}
        self.label_token_counts = {}
        self.label_lengths = {}
        self.trie = Trie()

        if language == 'en':
            import en_core_web_md
            self.spacy = en_core_web_md.load()
        elif language == 'fr':
            import fr_core_web_md
            self.spacy = fr_core_web_md.load()
        else:
            raise ValueError(f"Unsupported language: {language}")

    def _load_concept_labels(self, concept_id, labels):

        label_index = 0
        for label in labels:
            normalized = self.punctuation_remove.sub(" ", label).replace("-", " ").lower()
            tokens, _ = span_tokenize(self.spacy, normalized)
            # For each token
            key = str(concept_id) + ":::" + str(label_index)
            self.label_concept_index[normalized] = key
            self.label_token_counts[normalized] = len(tokens)
            self.label_lengths[normalized] = len(normalized)
            self.trie.add(normalized, count=1)
            label_index += 1

    def match_mentions(self, input_text) -> Tuple[List[Tuple[int, int]], List[str], Set[Annotation]]:
        normalized_text = self.punctuation_remove.sub(" ", input_text).replace("-", " ").lower()
        matches = []
        tokens, spans = span_tokenize(self.spacy, normalized_text)

        current_token_index = 0
        while current_token_index < len(tokens):
            current_match_cursor = 0
            while current_token_index + current_match_cursor < len(tokens):
                sub_string = normalized_text[spans[current_token_index][0]:spans[current_match_cursor][1]]
                found = self.trie.search_within_distance(sub_string, dist=2)
                if len(found) > 0:
                    # Register match
                    print(len(found))
                    current_match_cursor += 1
                else:
                    break
            current_token_index += 1

        return [], [], set()
