from typing import Set, Tuple, List
import torch.nn.functional as F

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from pyclinrec.dictionary import DictionaryLoader
from pyclinrec.recognizer import ConceptRecognizer, Concept, Annotation
from transformers import BitsAndBytesConfig

from torch.utils.data import Dataset, DataLoader
from pyclinrec.recognizer.recognizer import Annotation

from pyclinrec.utils.spacy_utils import span_tokenize

import scipy

from deeplake.core.vectorstore import VectorStore

import numpy as np

from pyclinrec.utils.stats import is_outlier, is_outlier_ecod


class _LabelDataset(Dataset):
    def __init__(self, labels, concept_ids) -> None:
        super().__init__()
        self.labels = labels
        self.concept_ids = concept_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if index < len(self.labels):
            return self.labels[index], self.concept_ids[index]
        else:
            None, None

    def get_concept_id(self, index):
        return self.concept_ids[index]


class _CandidateConcept:
    def __init__(self, concept_id) -> None:
        self.concept_id = concept_id
        self._candidate_mentions = []
        self._current_maximum = -3000

    def add_candidate_mention(
        self,
        text,
        concept_id,
        start_offset,
        end_offset,
        match_cursor,
        num_stopwords,
        similarity_score,
        z_score,
        embedding,
    ):
        mention_dict = {
            "text": text,
            "concept_id": concept_id,
            "start_offset": start_offset,
            "end_offset": end_offset,
            "match_cursor": match_cursor,
            "similarity_score": similarity_score,
            "num_stopwords": num_stopwords,
            "z_score": z_score,
            "embedding": embedding,
        }

        slope = 0
        if len(self._candidate_mentions) > 0:
            if z_score > self._current_maximum:
                self._current_maximum = z_score
                slope = 1
            elif z_score == self._current_maximum:
                slope = 0
            else:
                slope = -1

        mention_dict["slope"] = slope

        self._candidate_mentions.append(mention_dict)

    def find_best_candidate_mention(self):
        z_scores = [mention["z_score"] for mention in self._candidate_mentions]
        index_of_max = np.argmax(z_scores)
        return (
            self._candidate_mentions[index_of_max]
            if self._candidate_mentions[0]["slope"] >= 0
            and self._candidate_mentions[-1]["slope"] == -1
            else None
        )

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, _CandidateConcept):
            return self.concept_id == __value.concept_id
        return False

    def __hash__(self) -> int:
        return hash(self.concept_id)


class IntersEmbeddingConceptRecognizer(ConceptRecognizer):
    def __init__(
        self,
        dictionary_loader: DictionaryLoader,
        stop_words_file: str,
        termination_terms_file: str,
        language: str,
        model_name_or_path: str,
        batch_size=32,
        device="cpu",
        suffix="",
        filters=None,
    ):
        super().__init__(dictionary_loader, language, filters)

        self.concept_length_index = {}

        self.stop_words = self._load_word_list(stop_words_file)
        self.termination_terms = self._load_word_list(termination_terms_file)

        self.batch_size = batch_size
        self.device = device

        if device == "cpu":
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=False,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if device == "cpu":
            self.model = AutoModel.from_pretrained(
                model_name_or_path, quantization_config=nf4_config
            )
        else:
            self.model = AutoModel.from_pretrained(
                model_name_or_path,
                quantization_config=nf4_config,
                trust_remote_code=True,
                device_map={"": 0},
            )

        self.unk_token_id = self.tokenizer.unk_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        # self.stop_words = self._piece_wise_tokenize_token_list(self.stop_words)
        # self.termination_terms = self._piece_wise_tokenize_token_list(
        #     self.termination_terms
        # )

        self.store = VectorStore(
            path=f"./pyclinrec_vector_store_{suffix}",
            read_only=False,
            # exec_option="compute_engine",
        )

        if language == "en":
            import en_core_web_md

            self.spacy = en_core_web_md.load()
        elif language == "fr":
            import fr_core_news_md

            self.spacy = fr_core_news_md.load()
        else:
            raise ValueError(f"Unsupported language: {language}")

    def initialize(self):
        print("Now loading the dictionary...")
        self.dictionary_loader.load()
        dictionary = self.dictionary_loader.dictionary  # type : List[DictionaryEntry]
        print("Now indexing the dictionary...")
        concept_labels = []
        concept_label_ids = []
        for entry in tqdm(list(dictionary), desc="Loading all labels"):
            # we split concept ids from labels
            # fields = line.split("\t")
            label = entry.label
            concept_id = entry.id

            concept_labels.append(label)
            concept_label_ids.append(concept_id)

            if entry.synonyms:
                for synonym in entry.synonyms:
                    concept_labels.append(synonym)
                    concept_label_ids.append(concept_id)
            # self._load_concept_labels(concept_id, labels)

        dataset = _LabelDataset(concept_labels, concept_label_ids)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        for labels, concept_ids in tqdm(dataloader, desc="Embedding all labels"):
            self._index_concept_labels(list(concept_ids), list(labels))

    def _piece_wise_tokenize_token_list(self, token_list):
        final_token_list = []

        for token in token_list:
            sub_tokens = self.tokenizer(token)
            final_token_list.append(sub_tokens[-1])

        return self.tokenizer.convert_tokens_to_ids(final_token_list)

    def _span_tokenize(self, text):
        output = self.tokenizer(text, return_offsets_mapping=True)
        spans = output["offset_mapping"]
        tokens = output["input_ids"]
        text_fragments = [text[span[0] : span[1]] for span in spans]
        return tokens, spans

    def _index_concept_labels(self, concept_ids, labels):
        inputs = self.tokenizer(
            labels,
            max_length=512,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.device)
        tokens = inputs["input_ids"]
        att_masks = inputs["attention_mask"]
        last_tokens = [(att_mask == 0).nonzero()[0].item() for att_mask in att_masks]
        metadata_records = []
        embeddings = []
        with torch.no_grad():
            model_output = self.model(**inputs)
            per_concept_label_indexes = {}

            last_hidden_state = model_output["last_hidden_state"].detach().cpu()
            # pooled_output = model_output["pooler_output"].detach().cpu()
            for vector_index in range(last_hidden_state.shape[0]):
                # token_vectors = last_hidden_state[vector_index, 0:last_tokens[vector_index] - 1, :]
                concept_id = concept_ids[vector_index]
                if concept_id not in self.concept_index:
                    self.concept_index[concept_id] = Concept(concept_id)
                concept = self.concept_index[concept_id]
                pooled_output = self._compute_fragment_embedding(
                    0, last_tokens[vector_index] - 1, model_output, vector_index
                )
                concept.add_label(labels[vector_index], label_embedding=pooled_output)
                embeddings.append(pooled_output.numpy())
                concept_token_count = 0
                if concept_id not in per_concept_label_indexes:
                    per_concept_label_indexes[concept_id] = 1
                else:
                    per_concept_label_indexes[concept_id] += 1

                key = (
                    f"{str(concept_id)}:::{str(per_concept_label_indexes[concept_id])}"
                )
                metadata_records.append({"concept_id": concept_id})

                last_token_index = last_tokens[vector_index] - 1
                if len(tokens.shape) > 1:
                    tokens = tokens[vector_index, :last_token_index].detach().cpu()
                else:
                    tokens = tokens[:last_token_index].detach().cpu()

                concept_token_count = len(tokens)
                self.concept_length_index[key] = concept_token_count

            self.store.add(text=labels, embedding=embeddings, metadata=metadata_records)

    def _tokens_to_spans(self, tokens, text: str, initial_start_offset=0):
        spans = []  # type: List[Tuple[int, int, str]]
        start_offset = initial_start_offset
        end_offset = initial_start_offset

        for current_token_index in range(len(tokens)):
            raw_token = tokens[current_token_index]
            if "#" in raw_token:
                raw_token = raw_token.replace("#", " ").strip()
            end_offset += len(raw_token)
            span = (start_offset, end_offset, text[start_offset:end_offset])
            spans.append(span)
            if current_token_index < len(tokens) - 1 and text[end_offset] == " ":
                offset = 0
                while (
                    end_offset + offset < len(text)
                    and text[end_offset + offset + 1] == " "
                ):
                    offset += 1
                start_offset = end_offset + offset + 1
                end_offset += offset + 1
            else:
                start_offset = end_offset
        return spans

    def _is_span_termination_token(self, token_span, text):
        return text[token_span[0] : token_span[1]].strip() in self.termination_terms

    def _trace_statistics_for_candidate_concepts(
        self, text, concept_scores, concept_embeddings, next_concepts
    ):
        scores = np.hstack(list(concept_scores.values()))
        description = scipy.stats.describe(scores)
        mn = description.minmax[0]
        mx = description.minmax[1]
        mean = description.mean
        skew = description.skewness
        kurt = description.kurtosis
        var = description.variance
        stdev = description.variance**0.5

        std_err = scipy.stats.sem(scores)
        iqr = scipy.stats.iqr(scores)

        outlying_concept_indexes, outlying_z_scores = is_outlier(scores)
        z = outlying_z_scores[0] if len(outlying_z_scores) > 0 else 0
        if z > 0:
            concept = next_concepts[outlying_concept_indexes[0]].split("/")[-1]
        else:
            concept = ""
        # geom_std = scipy.stats.gstd(scores)
        # entropy = scipy.stats.entropy(scores)
        # diff_entropy = scipy.stats.differential_entropy(scores)

        # embedding_matrix = np.vstack(concept_embeddings)
        # distances = 1 - scipy.spatial.distance.pdist(embedding_matrix, metric="cosine")

        # hist_scores = np.histogram(scores, bins=4)
        # hist_distances = np.histogram(distances, bins=4)

        # distance_description = scipy.stats.describe(distances)

        # dist_mean = distance_description.mean
        # dist_var = distance_description.variance
        # dist_skew = distance_description.skewness
        # dist_kurt = distance_description.kurtosis
        # rel_ent = scipy.stats.entropy(hist_scores[-1], hist_distances[-1])

        # print(
        #     f'"{text}", {mn:2.5f}, {mx:2.5f}, {mean: 2.5f},{dist_mean:2.5f}, {var:2.5f}, {dist_var:2.5f}, {stdev:2.5f}, {std_err:2.5f}, {kurt:2.5f}, {dist_kurt:2.5f}, {skew:2.5f}, {dist_skew:2.5f}, {iqr:2.5f}, {geom_std:2.5f}, {entropy:2.5f}, {diff_entropy:2.5f}, {rel_ent:2.5f}'
        # )
        print(
            f'"{text}", min={mn:2.5f}, max={mx:2.5f}, mu={mean: 2.5f}, V={var:2.5f}, sig={stdev:2.5f}, ste={std_err:2.5f}, k={kurt:2.5f}, sk={skew:2.5f}, iqr={iqr:2.5f}, z={z:2.5f} ({concept}))'
        )

    def _match_subsequence(
        self,
        input_text,
        token_spans,
        current_span,
        current_token_span_index,
        model_output,
    ) -> set[str]:
        concept_start = current_span[0]
        # For now we have matched a single terms, so currently the end position will be that of the current token
        concept_end = current_span[1]
        match_cursor = 0
        stop_count = 0
        candidate_concepts = {}
        while current_token_span_index + match_cursor < len(
            token_spans
        ) and not self._is_span_termination_token(
            token_spans[current_token_span_index + match_cursor], input_text
        ):
            # We get the next token and position span
            next_span = token_spans[current_token_span_index + match_cursor]
            next_token = input_text[next_span[0] : next_span[1]]
            #  if the token is in the termination list the matching process ends here

            if next_token.strip() not in self.stop_words:
                text_up_to_now = input_text[concept_start : next_span[1]]

                # We try to find matching concepts and compute the intersection with previously identified concepts

                (
                    next_concepts,
                    concept_scores,
                    mean_similarity,
                    embeddings,
                ) = self._lookup_concepts(
                    current_token_span_index,
                    current_token_span_index + match_cursor,
                    model_output,
                )

                self._trace_statistics_for_candidate_concepts(
                    text_up_to_now, concept_scores, embeddings, next_concepts
                )
                outlying_concept_indexes, outlying_z_scores = is_outlier(
                    np.hstack(list(concept_scores.values()))
                )

                is_outlier_ecod(np.hstack(list(concept_scores.values())), embeddings)
                next_concepts = {
                    next_concepts[i]: {"z_score": z_score, "embedding": embeddings[i]}
                    for i, (z_score) in zip(outlying_concept_indexes, outlying_z_scores)
                }

                candidate_concepts |= {
                    concept_id: _CandidateConcept(concept_id)
                    for concept_id in next_concepts
                    if concept_id not in candidate_concepts
                }

                for concept_id in next_concepts:
                    if concept_id in candidate_concepts:
                        candidate_concepts[concept_id].add_candidate_mention(
                            text=text_up_to_now,
                            concept_id=concept_id,
                            start_offset=concept_start,
                            end_offset=next_span[1],
                            match_cursor=match_cursor,
                            num_stopwords=stop_count,
                            similarity_score=concept_scores[concept_id],
                            z_score=next_concepts[concept_id]["z_score"],
                            embedding=next_concepts[concept_id]["embedding"],
                        )

                concept_end = next_span[1]
            else:
                stop_count += 1
            match_cursor += 1
        final_candidate_concepts = []
        for concept_candidate in candidate_concepts.values():
            best_candidate = concept_candidate.find_best_candidate_mention()
            if best_candidate is not None:
                final_candidate_concepts.append(best_candidate)
        return match_cursor, final_candidate_concepts

    def _match_mentions(
        self, input_text
    ) -> Tuple[List[Tuple[int, int]], List[str], Set[Annotation]]:
        annotations = []

        # We split the text into token spans (begin and end position from the start of the text)
        tokens, token_spans = self._span_tokenize(input_text)
        model_output = self.compute_embedding(input_text)

        # we iterate over tokens one by one until we reach the end of the text
        current_token_span_index = 1
        while current_token_span_index < len(token_spans):
            # we get the current token span
            current_span = token_spans[current_token_span_index]
            concept_start = current_span[0]
            concept_end = current_span[1]

            current_text = input_text[concept_start:concept_end]
            if not self._is_stop_or_termination_token(current_text.strip()):
                (match_cursor, final_candidate_concepts) = self._match_subsequence(
                    input_text,
                    token_spans,
                    current_span,
                    current_token_span_index,
                    model_output,
                )

                annotations.extend(
                    self._create_annotations(input_text, final_candidate_concepts)
                )
                current_token_span_index += 1
            else:
                current_token_span_index += 1
        return (
            token_spans,
            [input_text[span[0] : span[1]] for span in token_spans],
            annotations,
        )

    def _create_annotations(self, input_text, final_candidate_concepts):
        annotations = []
        for concept in final_candidate_concepts:
            matched_length = concept["match_cursor"] - concept["num_stopwords"]
            # We check that the matched length is equal to the number of tokens in the concept
            # if matched_length == self.concept_length_index[concept]:
            concept_id = concept["concept_id"]
            annotation = Annotation(
                concept_id=concept_id,
                start=concept["start_offset"],
                end=concept["end_offset"],
                matched_text=input_text[
                    concept["start_offset"] : concept["end_offset"]
                ],
                matched_length=matched_length,
                label_key=concept_id,
                concept=self.concept_index[concept_id],
            )
            annotations.append(annotation)

        return annotations

    def _lookup_concepts(self, start_token, end_token, model_output) -> Set[Concept]:
        fragment_embedding = self._compute_fragment_embedding(
            start_token, end_token, model_output
        )
        result = self.store.search(
            embedding=fragment_embedding,
            k=20,
            return_tensors="*",
            distance_metric="COS",
        )
        concept_scores = {}
        for i in range(len(result["metadata"])):
            concept_id = result["metadata"][i]["concept_id"]
            score = result["score"][i]
            concept_scores[concept_id] = score
        return (
            [record["concept_id"] for record in result["metadata"]],
            concept_scores,
            np.array(result["score"]).mean(),
            result["embedding"],
        )

    def _compute_fragment_embedding(
        self, start_token, end_token, model_output, batch_index=0
    ):
        token_vectors = (
            model_output["last_hidden_state"][batch_index, start_token:end_token, :]
            .detach()
            .cpu()
        )
        return torch.mean(token_vectors, dim=0)

    def compute_embedding(self, text):
        inputs = self.tokenizer(
            [text],
            max_length=512,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            return self.model(**inputs)

    def _root_function(self, token) -> str:
        return token

    def _is_stop_or_termination_token(self, token):
        return token in self.stop_words or token in self.termination_terms

    def _is_span_termination_token(self, token_span, text):
        return text[token_span[0] : token_span[1]].strip() in self.termination_terms
