import json
from abc import abstractmethod, ABC

import requests
from tqdm import tqdm

from dictionary import DictionaryLoader


class DictionaryEnrichment(ABC):
    def __init__(self, loader: DictionaryLoader):
        self.loader = loader  # type : DictionaryLoader

    @abstractmethod
    def enrich(self):
        pass


class BioportalDictionaryEnrichment(DictionaryEnrichment):
    def __init__(self, loader: DictionaryLoader, api_url: str, apikey: str, max_k: int = 100):
        super().__init__(loader)
        self.url = api_url
        self.apikey = apikey
        self.max_k = max_k

    def enrich(self):
        for concept_id in tqdm(self.loader.dictionary_index):
            entry = self.loader.entry_from_index(concept_id)
            label = entry.label
            json_results = requests.get(
                "{url}/search?q={label}&include_properties=true&require_definition="
                "fase&display_context=false&pagesize={maxk}&include=prefLabel,synonym,cui,semanticType&exact_match=false&apikey={apikey}".format(
                    label=label, apikey=self.apikey, url=self.url, maxk=self.max_k))

            parsed = json.loads(json_results.text)
            collection = parsed['collection']
            if len(collection) > 0:
                links = []
                cuis = set()
                tuis = set()
                synonyms = set()
                for item in collection:
                    links.append(item['@id'])
                    try:
                        cuis.update(item['cui'])
                        tuis.update(item['semanticType'])
                        synonyms.update(item['synonym'])
                    except KeyError:
                        pass

                entry.mappings = links
                entry.synonyms = synonyms
                entry.cuis = cuis
                entry.tuis = tuis
