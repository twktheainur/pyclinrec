from abc import ABC, abstractmethod
from typing import Set, Tuple, List

import jellyfish
import numpy
import regex
from nltk import StemmerI, SnowballStemmer

from pyclinrec.dictionary import DictionaryLoader
from pyclinrec.recognizer import ConceptRecognizer, Concept, Annotation, AnnotationFilter
from pyclinrec.utils.spacy_utils import span_tokenize


class IntersectionConceptRecognizer(ConceptRecognizer, ABC):

    def __init__(self, dictionary_loader: DictionaryLoader, stop_words_file: str, termination_terms_file: str,
                 language="en", filters: List[AnnotationFilter] = None):
        """
        Parameters
        ----------
            dictionary_loader: DictionaryLoader
                The dictionary loader that will provide the dictionary contents
            stop_words_file: str
                Path to a text file containing a list of stop words (one per line)
            termination_terms_file: str
                Path to a text file containing a list of termination terms that stop the production of additional
                tokens in a matching mention.
            language: str
                The language of the text that will processed (affects the choice of tokenner and stemmer).
            filters: List[AnnotationFilter]
                A list of filters to apply post recognition
        """
        super().__init__(dictionary_loader, language=language, filters=filters)
        self.stop_words = self._load_word_list(stop_words_file)
        self.termination_terms = self._load_word_list(termination_terms_file)
        self.unigram_root_index = dict()  # record the phone and give an Id
        self.concept_length_index = dict()  # record the phone and give the length of the expression
        self.unigram_concept_count_histogram = dict()

        if language == 'en':
            import en_core_web_md
            self.spacy = en_core_web_md.load()
        elif language == 'fr':
            import fr_core_news_md
            self.spacy = fr_core_news_md.load()
        else:
            raise ValueError(f"Unsupported language: {language}")

    @abstractmethod
    def _root_function(self, token) -> str:
        pass

    def _concept_from_root(self, root) -> Set[Concept]:
        if root not in self.unigram_root_index:
            return set()
        else:
            return self.unigram_root_index[root]

    def _load_concept_labels(self, concept_id, labels):
        punctuation_remove = regex.compile(r'[\p{C}|\p{M}|\p{P}|\p{S}|\p{Z}]+', regex.UNICODE)
        label_index = 0
        for label in labels:

            normalized = punctuation_remove.sub(" ", label).lower()
            # We tokenize the label
            tokens, _ = span_tokenize(self.spacy, normalized)

            concept_token_count = 0
            # For each token
            key = str(concept_id) + ":::" + str(label_index)
            for token in tokens:
                # We skip words that belong to the stop list and words that contain non alphanumerical characters
                if token not in self.stop_words:
                    token_phone = self._root_function(token)

                    # we create the dictionary entry if it did not exist before
                    if token_phone not in self.unigram_root_index:
                        self.unigram_root_index[
                            token_phone] = set()
                    # if it already existed we add the concept id to the corresponding set
                    self.unigram_root_index[token_phone].add(key)
                    count_key = (key, token_phone)
                    if count_key not in self.unigram_concept_count_histogram:
                        self.unigram_concept_count_histogram[count_key] = 1
                    else:
                        self.unigram_concept_count_histogram[count_key] = self.unigram_concept_count_histogram[
                                                                              count_key] + 1

                    concept_token_count += 1
            self.concept_length_index[key] = concept_token_count
            label_index += 1

    def match_mentions(self, input_text) -> Tuple[List[Tuple[int, int]], List[str], Set[Annotation]]:
        punctuation_remove = regex.compile(r'[\p{C}|\p{M}|\p{P}|\p{S}]+', regex.UNICODE)
        annotations = []

        # We normalize the text (Remove all punctuation and replace with whitespace)
        # normalized_input_text = punctuation_remove.sub(" ", input_text).replace("-", " ").lower()
        # normalized_input_text = input_text.replace("-", " ").lower()
        normalized_input_text = input_text

        # We split the text into token spans (begin and end position from the start of the text)
        tokens, token_spans = span_tokenize(self.spacy, normalized_input_text)

        # we iterate over tokens one by one until we reach the end of the text
        current_token_span_index = 0
        while current_token_span_index < len(token_spans):
            # we get the current token span
            current_span = token_spans[current_token_span_index]

            # we extract the string of the token from the text
            token = normalized_input_text[current_span[0]:current_span[1]]

            # if the word is a stoplist term or a termination term we skip it
            if token not in self.stop_words and token not in self.termination_terms:
                # We get the concept ids matching the phone of the current token
                token_root = self._root_function(token)
                concepts = self._concept_from_root(token_root)

                # this is the start position of the first token of a matching sequence
                concept_start = current_span[0]
                # For now we have matched a single terms, so currently the end position will be that of the current
                # token
                concept_end = current_span[1]
                match_cursor = 1
                stop_count = 0
                while current_token_span_index + match_cursor < len(token_spans):

                    # We get the next token and position span
                    next_span = token_spans[current_token_span_index + match_cursor]
                    next_token = normalized_input_text[next_span[0]:next_span[1]]

                    # if the token is in the termination list the matching process ends here
                    if next_token in self.termination_terms:
                        break
                    # If the token is in the Stop list we skip it and increment the count of the skipped words
                    # We will need to subtract this from the total number of tokens for the concept
                    elif next_token in self.stop_words:
                        stop_count += 1
                    # Otherwise we try to find a match for the token phone in the dictionary index
                    else:
                        # we doublemetaphone the token's text
                        next_token_phone = self._root_function(next_token)

                        # We try to find matching concepts and compute the intersection with previously identified
                        # concepts

                        next_concepts = self._concept_from_root(next_token_phone) & concepts

                        # if we find none we stop the matching here
                        if len(next_concepts) == 0:
                            break
                        else:
                            # Filtering for duplicate words if concept doesn't also contain duplication
                            # Otherwise drug-drug in drug-drug interaction would match any concept with a two
                            # token label where drug appears only once in the label.
                            # e.g. drug resistant would match on drug-drug.
                            filtered_next_concepts = set()
                            for concept_key in next_concepts:
                                if not (next_token_phone == token_root and
                                        self.unigram_concept_count_histogram[(concept_key, token_root)] < 2):
                                    filtered_next_concepts.add(concept_key)

                            # if we find a match, then we update the current end position to that of the currently
                            # matching token and update the intersected matched concept buffer
                            concepts = next_concepts if len(filtered_next_concepts) else filtered_next_concepts
                            concept_end = next_span[1]

                    # if we arrive here the current token has matched, we keep count of the current match length
                    match_cursor += 1

                # Once we get out of the loop we reconstruct the matches from the concepts remaining in the set
                # after successive intersections, if concepts is empty there was no match and so
                # Tokens.conceptsToAnnotationTokens will return an empty list otherwise we get a list of
                # AnnotationToken objects instances that we add to the list of identified concepts

                for concept in concepts:
                    key_parts = concept.split(":::")
                    concept_id = str(key_parts[0])

                    annotation = Annotation(concept_id, concept_start, concept_end,
                                            input_text[concept_start:concept_end],
                                            match_cursor - stop_count, label_key=concept,
                                            concept=self.concept_index[concept_id])
                    annotations.append(annotation)

            current_token_span_index += 1
            # Here we filter the annotations to keep only those where the concept length matches the length of the
            # identified annotation

        return token_spans, [normalized_input_text[span[0]:span[1]] for span in token_spans], set(
            [annotation for annotation in annotations if
             annotation.matched_length == self.concept_length_index[annotation.label_key]
             ]
        )


class LevenshteinAnnotationFilter(AnnotationFilter):

    def __init__(self, theta=0.85):
        super().__init__()
        self.theta = theta

    def apply_filter(self, annotations: Set[Annotation], text, tokens_spans, tokens) -> Set[Annotation]:
        final_annotations = set()
        for annotation in annotations:
            if annotation.matched_length > 1:
                final_annotations.add(annotation)
            elif annotation.matched_length == 1 \
                    and self._max_levenshtein_less_than_theta(annotation.matched_text, annotation.concept):
                final_annotations.add(annotation)

        return final_annotations

    def _max_levenshtein_less_than_theta(self, mention: str, concept: Concept):
        distances = []
        for label in concept.labels:
            distances.append(1 - jellyfish.damerau_levenshtein_distance(mention, label))
        return numpy.array(distances).max(0) > self.theta


# class InterDoubleMetaphoneConceptRecognizer(IntersectionConceptRecognizer):
#
#     def __init__(self, dictionary_loader: DictionaryLoader, stop_words_file: str, termination_terms_file: str,
#                  language="en"):
#         super().__init__(dictionary_loader, stop_words_file, termination_terms_file, language,
#                          [LevenshteinAnnotationFilter()])
#
#     def _root_function(self, token) -> str:
#         return doublemetaphone(token)[0]


class IntersStemConceptRecognizer(IntersectionConceptRecognizer):

    def __init__(self, dictionary_loader: DictionaryLoader, stop_words_file: str, termination_terms_file: str,
                 language="en",
                 stemmer: StemmerI = None,
                 filters=None):
        """
        Constructs an InterStemConceptRecognizer instance.
            Parameters
             ----------
                 dictionary_loader: DictionaryLoader
                     The dictionary loader that will provide the dictionary contents
                 stop_words_file: str
                     Path to a text file containing a list of stop words (one per line)
                 termination_terms_file: str
                     Path to a text file containing a list of termination terms that stop the production of additional
                     tokens in a matching mention.
                 language: str
                     The language of the text that will processed (affects the choice of tokenner and stemmer).
                     Default: en, Supported en,fr
                 stemmer: StemmerI
                     The stemmer to use, especially if lang is different from 'en' or 'fr'
                 filters: List[AnnotationFilter]
                    Override default annotation filters
        """
        super().__init__(dictionary_loader, stop_words_file, termination_terms_file, language,
                         filters=filters if filters is not None else [LevenshteinAnnotationFilter(theta=0.85)])

        if not stemmer:
            if self.language == "en":
                self.stemmer = SnowballStemmer("english")
            else:
                self.stemmer = SnowballStemmer("french")
        else:
            self.stemmer = stemmer

        self.punctuation_remove = regex.compile(r'\p{C}', regex.UNICODE)

    def _root_function(self, token) -> str:
        return self.stemmer.stem(self.stemmer.stem(token))
