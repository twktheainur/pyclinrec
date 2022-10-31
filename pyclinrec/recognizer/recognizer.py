from abc import ABC, abstractmethod
from typing import List, Set, Dict, Tuple

import torch
from tqdm import tqdm


class Concept:
    def __init__(self, uri, labels: Set[str] = None, label_embeddings: Dict[str, torch.tensor] = None,
                 definition: str = ""):
        self.label_embeddings = label_embeddings
        if label_embeddings is None:
            self.label_embeddings = dict()
        self.uri = uri
        if labels is not None:
            self.labels = labels
        else:
            self.labels = set()
        self.definition = definition

    def add_label(self, label, label_embedding: torch.tensor = None):
        self.labels.add(label)
        if label_embedding is not None:
            self.label_embeddings[label] = label_embedding

    def __eq__(self, other):
        return self.uri == other.uri

    def __str__(self):
        return "Concept [" + ", ".join(self.labels) + " | " + self.definition + " ]"


class Annotation:
    def __init__(self, concept_id, start, end, matched_text, matched_length, label_key: str = None,
                 concept: Concept = None):
        self.concept_id = concept_id
        if label_key:
            self.label_key = label_key
        else:
            self.label_key = concept_id

        self.start = start
        self.end = end
        self.matched_text = matched_text
        self.matched_length = matched_length
        self.loaded_concept = None
        self.confidence_score = 1
        self.concept = concept

    def __str__(self) -> str:
        return "Annotation ({})[{}-{} M{}] - {}".format(self.matched_text, self.start, self.end, self.matched_length,
                                                        self.concept_id)

    def __eq__(self, o) -> bool:
        if isinstance(o, Annotation):
            return 'concept_id' in o.__dict__ and self.concept_id == o.concept_id and self.start == o.start and self.end == o.end
        else:
            return False

    def __hash__(self) -> int:
        return hash((self.matched_text, self.start, self.end, self.concept_id))


class ConceptIndexer(ABC):
    def __init__(self):
        pass


class AnnotationFilter(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def apply_filter(self, annotations: Set[Annotation], text, tokens_spans: List[Tuple[int, int]], tokens) -> Set[
        Annotation]:
        """
        Filter the annotations provided and return a list of filtered annotations of length lesser or equal to the
        annotations provided.
        Parameters
        ----------
            annotations: List[Annotation]
                The list of annotations to filter
            text: str
                The original text that corresponds to the annotations
            tokens_spans: Tuple[int,int]
                The spans of each token of the tokenized text
            tokens: List[str]
                Tokens corresponding to the spans in the text

        Return
        ------
            filtered_annotations: List[Annotation]
                The list of filtered annotations
        """
        pass


class ConceptRecognizer(ABC):
    def __init__(self, dictionary_loader, language="en",
                 filters: List[AnnotationFilter] = None):
        """
        This is the constructor of an Abstract class and should never be called directly, see subclasses.
             Parameters
             ----------
                 dictionary_loader: DictionaryLoader
                     The dictionary loader that will provide the dictionary contents
                 language: str
                     The language of the text that will processed (affects the choice of tokenizer and stemmer).
                 filters: List[AnnotationFilter]
                     A list of filters to apply post recognition
             """
        self.dictionary_loader = dictionary_loader
        self.concept_index = dict()  # type: Dict[str,Concept]
        self.language = language
        if filters is None:
            filters = []
        self.filters = filters

    @staticmethod
    def _load_word_list(file) -> List[str]:
        with open(file, "r", encoding="utf8") as word_file:
            words = [line.strip() for line in word_file.readlines()]
        return words

    @abstractmethod
    def _load_concept_labels(self, concept_id, labels):
        pass

    def initialize(self):
        print("Now loading the dictionary...")
        self.dictionary_loader.load()
        dictionary = self.dictionary_loader.dictionary  # type : List[DictionaryEntry]
        print("Now indexing the dictionary...")
        for entry in tqdm(dictionary):
            # we split concept ids from labels
            # fields = line.split("\t")
            label = entry.label
            concept_id = entry.id

            labels = [label]
            if entry.synonyms:
                labels.extend(entry.synonyms)
            concept = Concept(concept_id, set(labels))
            self.concept_index[concept_id] = concept
            self._load_concept_labels(concept_id, labels)

    @abstractmethod
    def match_mentions(self, input_text) -> Tuple[List[Tuple[int, int]], List[str], Set[Annotation]]:
        """Match candidate mentions of entities from the dictionary in the text
        Parameters
        ----------
            input_text: str
                The input text on which mentions should be matched
        Returns
        -------
            token_spans: List[Tuple[int, int]]
                The token spans for the input_text used as the basis for mention annotation.
            tokens: List[str]
                The corresponding tokens
            annotations: Set[Annotation]
                Mention annotations, see @Annotation
        """

        raise NotImplementedError("This abstract method must be overridden")

    def annotate(self, input_text) -> Tuple[List[Tuple[int, int]], List[str], Set[Annotation]]:
        """Matches candidate mentions of entities from the dictionary in the text and
        applies pre and post-processing filters
        Parameters
        ----------
            input_text: str
                The input text on which mentions should be matched
        Returns
        -------
            token_spans: List[Tuple[int, int]]
                The token spans for the input_text used as the basis for mention annotation.
            tokens: List[str]
                The corresponding tokens
            annotations: Set[Annotation]
                Mention annotations, see @Annotation
        """
        token_spans, tokens, annotations = self.match_mentions(input_text)
        for annotation_filter in self.filters:
            annotations = annotation_filter.apply_filter(annotations, input_text, token_spans, tokens)
        return token_spans, tokens, annotations
