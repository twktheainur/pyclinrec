import re
from typing import Set, Tuple, List

import regex
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig

from pyclinrec.dictionary import DictionaryLoader
from pyclinrec.recognizer import ConceptRecognizer, Concept, Annotation
from pyclinrec.recognizer.intersection_recognizers import IntersectionConceptRecognizer
from transformers import BitsAndBytesConfig

from torch.utils.data import Dataset, DataLoader

class _LabelDataset(Dataset):
    def __init__(self, labels, concept_ids) -> None:
        super().__init__()
        self.labels = labels
        self.concept_ids = concept_ids
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.labels[index], self.concept_ids[index]
    
    def get_concept_id(self, index):
        return self.concept_ids[index]
        

class IntersEmbeddingConceptRecognizer(IntersectionConceptRecognizer):

    def __init__(self, dictionary_loader: DictionaryLoader, stop_words_file: str, termination_terms_file: str, language: str,
                model_name_or_path: str, batch_size=32, device="cpu"):
        super().__init__(dictionary_loader, stop_words_file, termination_terms_file, language)
        self.concept_token_vector_index = {}
        self.batch_size = batch_size
        self.device = device
        
        if device == "cpu":
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=False,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if device == "cpu":
            self.model = AutoModel.from_pretrained(model_name_or_path, quantization_config=nf4_config)
        else: 
            self.model = AutoModel.from_pretrained(model_name_or_path, quantization_config=nf4_config, trust_remote_code=True, device_map={"":0})
            
        self.unk_token_id = self.tokenizer.unk_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.stop_words = self._piece_wise_tokenize_token_list(self.stop_words)
        self.termination_terms = self._piece_wise_tokenize_token_list(self.termination_terms)

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
                concept_labels.extend(entry.synonyms)
                concept_label_ids.append(concept_id)
            # self._load_concept_labels(concept_id, labels)
        
        dataset = _LabelDataset(concept_labels, concept_label_ids)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
        for labels, concept_ids in tqdm(dataloader, desc="Embedding all labels"):
            self._embed_batch_concept_labels(list(concept_ids), list(labels))

    def _piece_wise_tokenize_token_list(self, token_list):
        final_token_list = []

        for token in token_list:
            sub_tokens = self.tokenizer.tokenize(token)
            final_token_list.append(sub_tokens[-1])

        return self.tokenizer.convert_tokens_to_ids(final_token_list)

    def _embed_batch_concept_labels(self, concept_ids, labels):

        inputs = self.tokenizer(labels, max_length=512, padding='max_length', return_attention_mask=True, return_tensors='pt').to(self.device)
        tokens = inputs['input_ids']
        att_masks = inputs['attention_mask']
        last_tokens = [(att_mask == 0).nonzero()[0].item() for att_mask in att_masks]
        with torch.no_grad():
            model_output = self.model(**inputs)
            per_concept_label_indexes = {}
            
            last_hidden_state = model_output['last_hidden_state'].detach().cpu()

            for vector_index in range(last_hidden_state.shape[0]):
            
                token_vectors = last_hidden_state[vector_index, 0:last_tokens[vector_index] - 1, :]
                concept_id = concept_ids[vector_index]
                if concept_id not in self.concept_index:
                    self.concept_index[concept_id] = Concept(concept_id)
                concept = self.concept_index[concept_id]
                concept.add_label(labels[vector_index], label_embedding=model_output['pooler_output'].detach().cpu())
                concept_token_count = 0
                if concept_id not in per_concept_label_indexes:
                    per_concept_label_indexes[concept_id] = 1
                else:
                    per_concept_label_indexes[concept_id] += 1
                key = f"{str(concept_id)}:::{str(per_concept_label_indexes[concept_id])}"
                last_token_index = last_tokens[vector_index] - 1
                if len(tokens.shape) > 1:
                    tokens = tokens[vector_index,:last_token_index].detach().cpu()
                else:
                    tokens = tokens[:last_token_index].detach().cpu()
                for token in tokens:
                    token = token.item()
                    # We skip words that belong to the stop list and words that contain non alphanumerical characters
                    # we create the dictionary entry if it did not exist before
                    if token not in self.unigram_root_index:
                        self.unigram_root_index[token] = set()
                    # if it already existed we add the concept id to the corresponding set
                    self.unigram_root_index[token].add(key)

                    if token not in self.concept_token_vector_index:
                        self.concept_token_vector_index[token] = {}
                    self.concept_token_vector_index[token][concept_id] = token_vectors[concept_token_count, :]
                    concept_token_count += 1
                self.concept_length_index[key] = concept_token_count

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
                while end_offset + offset < len(text) and text[end_offset + offset + 1] == " ":
                    offset += 1
                start_offset = end_offset + offset + 1
                end_offset += offset + 1
            else:
                start_offset = end_offset
        return spans

    def _concept_from_root(self, root) -> Set[Concept]:
        if root not in self.unigram_root_index:
            return set()
        else:
            return self.unigram_root_index[root]
        
    def _root_function(self, token) -> str:
        return token