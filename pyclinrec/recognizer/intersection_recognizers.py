from abc import ABC, abstractmethod
from typing import Set, Tuple, List

import jellyfish
import numpy
from nltk import StemmerI, SnowballStemmer

from pyclinrec.dictionary import DictionaryLoader
from pyclinrec.recognizer import (
    ConceptRecognizer,
    Concept,
    Annotation,
    AnnotationFilter,
)
from pyclinrec.utils.re import PUNCTUATION_REGEX
from pyclinrec.utils.spacy_utils import span_tokenize


class IntersectionConceptRecognizer(ConceptRecognizer, ABC):
    def __init__(
        self,
        dictionary_loader: DictionaryLoader,
        stop_words_file: str,
        termination_terms_file: str,
        language="en",
        filters: List[AnnotationFilter] = None,
    ):
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
                The language of the text that will processed (affects the choice of tokenizer and stemmer).
            filters: List[AnnotationFilter]
                A list of filters to apply post recognition
        """
        
        super().__init__(dictionary_loader, language=language, filters=filters)
        self.stop_words = self._load_word_list(stop_words_file)
        self.termination_terms = self._load_word_list(termination_terms_file)
        self.unigram_root_index = {}
        self.concept_length_index = {}
        self.unigram_concept_count_histogram = {}

        if language == "en":
            import en_core_web_md

            self.spacy = en_core_web_md.load()
        elif language == "fr":
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

    def _embed_batch_concept_labels(self, concept_id, labels):
        for label_index, label in enumerate(labels):
            normalized = PUNCTUATION_REGEX.sub(" ", label).lower()
            # We tokenize the label
            tokens, _ = span_tokenize(self.spacy, normalized)

            concept_token_count = 0
            # For each token
            key = f"{str(concept_id)}:::{str(label_index)}"
            for token in tokens:
                # We skip words that belong to the stop list and words that contain non alphanumerical characters
                if token not in self.stop_words:
                    token_phone = self._root_function(token)

                    # we create the dictionary entry if it did not exist before
                    if token_phone not in self.unigram_root_index:
                        self.unigram_root_index[token_phone] = set()
                    # if it already existed we add the concept id to the corresponding set
                    self.unigram_root_index[token_phone].add(key)
                    count_key = (key, token_phone)
                    self.unigram_concept_count_histogram[count_key] = (
                        1
                        if count_key not in self.unigram_concept_count_histogram
                        else self.unigram_concept_count_histogram[count_key] + 1
                    )
                    concept_token_count += 1
            self.concept_length_index[key] = concept_token_count

    def _match_subsequence(
        self,
        input_text,
        token_spans,
        current_span,
        current_token_span_index,
        token_root,
        concepts,
    ) -> set[str]:
        concept_start = current_span[0]
        # For now we have matched a single terms, so currently the end position will be that of the current token
        concept_end = current_span[1]
        match_cursor = 1
        stop_count = 0
        while (
            current_token_span_index + match_cursor < len(token_spans)
            and not self._is_span_termination_token(token_spans[current_token_span_index + match_cursor], input_text)
        ):
            # We get the next token and position span
            next_span = token_spans[current_token_span_index + match_cursor]
            next_token = input_text[next_span[0] : next_span[1]]

            #  if the token is in the termination list the matching process ends here

            if next_token not in self.stop_words:
                next_token_root = self._root_function(next_token)

                # We try to find matching concepts and compute the intersection with previously identified concepts
                next_concepts = self._concept_from_root(next_token_root) & concepts

                filtered_next_concepts = self._filter_next_concepts(
                    token_root, next_token_root, next_concepts
                )
                if len(filtered_next_concepts) == 0:
                    break
                concept_end = next_span[1]
                concepts = (
                    next_concepts
                    if len(filtered_next_concepts)
                    else filtered_next_concepts
                )

            else:
                stop_count += 1
            match_cursor += 1
        return concept_start, concept_end, match_cursor, stop_count, concepts

    def _filter_next_concepts(self, token_root, next_token_root, next_concepts):
        return {
            concept_key
            for concept_key in next_concepts
            if next_token_root != token_root
            or self.unigram_concept_count_histogram[(concept_key, token_root)] >= 2
        }

    def _match_mentions(
        self, input_text
    ) -> Tuple[List[Tuple[int, int]], List[str], Set[Annotation]]:
        annotations = []

        # We split the text into token spans (begin and end position from the start of the text)
        tokens, token_spans = span_tokenize(self.spacy, input_text)

        # we iterate over tokens one by one until we reach the end of the text
        current_token_span_index = 0
        while current_token_span_index < len(token_spans):
            # we get the current token span
            current_span = token_spans[current_token_span_index]

            token_text = input_text[current_span[0] : current_span[1]]

            # if the word is a stop list term or a termination term we skip it
            if not self._is_stop_or_termination_token(token_text):
                # We get the concept ids matching the root of the current token
                token_root = self._root_function(token_text)
                concepts = self._concept_from_root(token_root)

                (
                    concept_start,
                    concept_end,
                    match_cursor,
                    stop_count,
                    concepts,
                ) = self._match_subsequence(
                    input_text,
                    token_spans,
                    current_span,
                    current_token_span_index,
                    token_root,
                    concepts,
                )

                # Once we get out of the loop we reconstruct the matches from the concepts remaining in the set
                # after successive intersections, if concepts is empty there was no match and so
                # Tokens.conceptsToAnnotationTokens will return an empty list otherwise we get a list of
                # AnnotationToken objects instances that we add to the list of identified concepts

                annotations.extend(
                    self._create_annotations(
                        input_text,
                        concepts,
                        concept_start,
                        concept_end,
                        match_cursor,
                        stop_count,
                    )
                )

            current_token_span_index += 1

        return (
            token_spans,
            [input_text[span[0] : span[1]] for span in token_spans],
            annotations,
        )

    def _create_annotations(
        self,
        input_text,
        concepts,
        concept_start,
        concept_end,
        match_cursor,
        stop_count,
    ):
        annotations = []
        for concept in concepts:
            matched_length =  match_cursor - stop_count
            # We check that the matched length is equal to the number of tokens in the concept
            if matched_length == self.concept_length_index[concept]:
                key_parts = concept.split(":::")
                concept_id = str(key_parts[0])
                annotation = Annotation(
                    concept_id = concept_id,
                    start=concept_start,
                    end=concept_end,
                    matched_text = input_text[concept_start:concept_end],
                    matched_length=matched_length,
                    label_key=concept,
                    concept=self.concept_index[concept_id],
                )
                annotations.append(annotation)

        return annotations

    def _is_stop_or_termination_token(self, token):
        return token in self.stop_words or token in self.termination_terms

    def _is_span_termination_token(self, token_span, text):
        return text[token_span[0]:token_span[1]].strip() in self.termination_terms

class LevenshteinAnnotationFilter(AnnotationFilter):
    def __init__(self, theta=0.85):
        super().__init__()
        self.theta = theta

    def apply_filter(
        self, annotations: Set[Annotation], text, tokens_spans, tokens
    ) -> Set[Annotation]:
        final_annotations = set()
        for annotation in annotations:
            if annotation.matched_length > 1:
                final_annotations.add(annotation)
            elif (
                annotation.matched_length == 1
                and self._max_levenshtein_less_than_theta(
                    annotation.matched_text, annotation.concept
                )
            ):
                final_annotations.add(annotation)

        return final_annotations

    def _max_levenshtein_less_than_theta(self, mention: str, concept: Concept):
        distances = [
            1 - jellyfish.damerau_levenshtein_distance(mention, label)
            for label in concept.labels
        ]
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
    def __init__(
        self,
        dictionary_loader: DictionaryLoader,
        stop_words_file: str,
        termination_terms_file: str,
        language="en",
        stemmer: StemmerI = None,
        filters=None,
    ):
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
        super().__init__(
            dictionary_loader,
            stop_words_file,
            termination_terms_file,
            language,
            filters=filters
            if filters is not None
            else [LevenshteinAnnotationFilter(theta=0.85)],
        )

        if stemmer:
            self.stemmer = stemmer

        elif self.language == "en":
            self.stemmer = SnowballStemmer("english")
        else:
            self.stemmer = SnowballStemmer("french")
        self.punctuation_remove = PUNCTUATION_REGEX

    def _root_function(self, token) -> str:
        return self.stemmer.stem(self.stemmer.stem(token))
