from abc import ABC, abstractmethod
from logging import getLogger
from typing import List

import pandas
from tqdm import tqdm



logger = getLogger("Dictionaries")


class DictionaryEntry:
    def __init__(self, id: str, label: str, definition: str = None, source: str = None, language: str = None, 
                mappings: List[str] = None, 
                cuis: List[str] = None, 
                tuis: List[str] = None, 
                synonyms: List[str] = None):
        """
            Class to represent a dictionary entry.

            Parameters
            ----------
            id : int
                Identifier of the dictionary entry
            label : str
                Label for the concept

        """
        self.id = id
        self.label = label
        self.synonyms = synonyms
        self.definition = definition
        self.source = source
        self.language = language
        self.mappings = mappings
        self.cuis = cuis
        self.tuis = tuis

    def __str__(self):
        return "{[{id}] {label}}".format(id=self.id, label=self.label)


class DictionaryLoader(ABC):
    def __init__(self, dictionary_file):
        self.dictionary = []  # type: List[DictionaryEntry]
        self.dictionary_index = {}  # type : Dict[int, DictionaryEntry]
        self.dictionary_file = dictionary_file
        self.reverse_index = {}

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def save(self, output_file: str):
        pass

    def entry_from_index(self, id: str) -> DictionaryEntry:
        """
        Get an entry from its index

        Returns
        -------
        dict_entry : DictionaryEntry
            The corresponding dictionary entry
        """
        return self.dictionary_index[id]

    def index(self, label: str):
        return self.reverse_index[label]

    def size(self):
        return len(self.dictionary_index.values())

    def __iter__(self):
        return self.dictionary_index.__iter__()


class MgrepDictionaryLoader(DictionaryLoader):

    def load(self):
        data = pandas.read_csv(self.dictionary_file, delimiter="\t", encoding="utf8", dtype=str)
        for index, row in tqdm(data.iterrows()):
            cid = row[0]
            if cid in self.dictionary_index.keys():
                entry = self.dictionary_index[cid]  # type : DictionaryEntry
                synonyms = entry.synonyms
                if not synonyms:
                    entry.synonyms = []
                entry.synonyms.append(row[1])
            else:
                entry = DictionaryEntry(cid, row[1])
                self.dictionary.append(entry)
                self.dictionary_index[cid] = entry
                self.reverse_index[row[1]] = cid

    def save(self, output_file: str):
        with open(output_file, "w", encoding="utf8") as output:
            for key, value in self.dictionary_index.items():
                output.write("{id}\t{label}\n".format(id=key, label=value.label))
                if value.synonyms:
                    for synonym in value.synonyms:
                        output.write("{id}\t{label}\n".format(id=key, label=synonym))
            output.flush()


class StringDictionaryLoader(MgrepDictionaryLoader):

    def __init__(self, string_entries):
        super().__init__(None)
        self.dictionary_string_entries = string_entries

    def load(self):
        for string_entry in self.dictionary_string_entries:
            ident = string_entry[0]
            if ident in self.dictionary_index.keys():
                entry = self.dictionary_index[ident]  # type : DictionaryEntry
                synonyms = entry.synonyms
                if not synonyms:
                    entry.synonyms = []
                entry.synonyms.append(string_entry[1])
            else:
                entry = DictionaryEntry(ident, string_entry[1])
                self.dictionary.append(entry)
                self.dictionary_index[ident] = entry
                self.reverse_index[string_entry[1]] = ident


def generate_brat_normalization_database(string_entries, target_file="brat_norm_db.txt", remove_uris=True):
    concept_dictionary = {}
    for item in string_entries:
        key = item[0].split("/")[-1].split("_")[1] if remove_uris else item[0]
        if key not in concept_dictionary:
            concept_dictionary[key] = []
        concept_dictionary[key].append(item[1])
    with open(target_file, "w") as target_handler:
        for key, values in concept_dictionary.items():
            entry = f"{key}"
            for value in values:
                entry += f"\tname:Name:{value}"
            entry += '\n'
            target_handler.write(entry)


def generate_dictionary_from_skos_file(self, graph = None, thesaurus_path=None,
                                        save_file="agrovoc_dictionary.tsv",
                                        skos_xl_labels=True,
                                        lang="fr"):
    from rdflib import Namespace, Graph
    self.graph = Graph() if graph is None else graph
    logger.info(f"Loading thesaurus... [{thesaurus_path}]")

    self.graph.load(thesaurus_path)

    if skos_xl_labels:
        query = f"""SELECT ?x ?lf WHERE {{
            ?x a skos:Concept;
            skosxl:prefLabel ?l.
            ?l skosxl:literalForm ?lf.
            FILTER(lang(?lf)='{lang}')
        }}
        """
        pref_labels = self.graph.query(query, initNs={'skos': Namespace("http://www.w3.org/2004/02/skos/core#"),
                                                    'skosxl': Namespace("http://www.w3.org/2008/05/skos-xl#")})
    else:
        query = f"""SELECT ?x ?lf WHERE {{
            ?x a skos:Concept;
            skos:prefLabel ?lf.
            FILTER(lang(?lf)='{lang}')
        }}
        """
        pref_labels = self.graph.query(query, initNs=dict(skos=Namespace("http://www.w3.org/2004/02/skos/core#")))

    string_entries = [(str(result[0]), str(result[1])) for result in pref_labels]
    if skos_xl_labels:
        query = f"""SELECT ?x ?lf WHERE {{
            ?x a skos:Concept;
            skosxl:prefLabel ?l.
            ?l skosxl:literalForm ?lf.
            FILTER(lang(?lf)='{lang}')
        }}
    """
        alt_labels = self.graph.query(query, initNs=dict(skos=Namespace("http://www.w3.org/2004/02/skos/core#"),
                                                        skosxl=Namespace("http://www.w3.org/2008/05/skos-xl#")))
    else:
        query = f"""SELECT ?x ?lf WHERE {{
        ?x a skos:Concept;
        skos:altLabel ?lf.
        FILTER(lang(?lf)='{lang}')
    }}
    """
        alt_labels = self.graph.query(query, initNs=dict(skos=Namespace("http://www.w3.org/2004/02/skos/core#")))

    string_entries.extend(
        (str(result[0]), str(result[1])) for result in alt_labels
    )
    generate_brat_normalization_database(string_entries, remove_uris=True)
    dictionary_loader = StringDictionaryLoader(string_entries)
    dictionary_loader.load()
    dictionary_loader.save(save_file)


def generate_dictionary_from_skos_sparql(endpoint,
                                        save_file="agrovoc_dictionary.tsv",
                                        skos_xl_labels=True,
                                        lang="fr", from_statement=""):
    from pyclinrec.utils.sparql import SparQLOffsetFetcher
    from SPARQLWrapper import SPARQLWrapper
    sparql = SPARQLWrapper(endpoint)

    if skos_xl_labels:
        fetcher = SparQLOffsetFetcher(sparql, 9000,
                                    where_body=f"?x a skos:Concept;\nskosxl:prefLabel ?l.\n ?l skosxl:literalForm ?lf.\nFILTER(lang(?lf)='{lang}')",
                                    select_columns="?x ?lf",
                                    prefixes="prefix skos: <http://www.w3.org/2004/02/skos/core#>\n"
                                            "prefix skosxl: <http://www.w3.org/2008/05/skos-xl#>",
                                    from_statement=from_statement)
    else:
        fetcher = SparQLOffsetFetcher(sparql, 9000,
                                    where_body=f"?x a skos:Concept;\nskos:prefLabel ?lf.\nFILTER(lang(?lf)='{lang}')",
                                    select_columns="?x ?lf",
                                    prefixes="prefix skos: <http://www.w3.org/2004/02/skos/core#>",
                                    from_statement=from_statement)
    results = fetcher.fetch_all()
    string_entries = [
        (result["x"]["value"], result["lf"]["value"]) for result in results
    ]
    if skos_xl_labels:
        fetcher = SparQLOffsetFetcher(sparql, 9000,
                                    where_body=f"?x a skos:Concept;\nskosxl:altLabel ?l.\n ?l skosxl:literalForm ?lf.\nFILTER(lang(?lf)='{lang}')",
                                    select_columns="?x ?lf",
                                    prefixes="prefix skos: <http://www.w3.org/2004/02/skos/core#>\n"
                                            "prefix skosxl: <http://www.w3.org/2008/05/skos-xl#>",
                                    from_statement=from_statement)
    else:
        fetcher = SparQLOffsetFetcher(sparql, 9000,
                                    where_body=f"?x a skos:Concept;\nskos:altLabel ?lf.\nFILTER(lang(?lf)='{lang}')",
                                    select_columns="?x ?lf",
                                    prefixes="prefix skos: <http://www.w3.org/2004/02/skos/core#>",
                                    from_statement=from_statement)
    results = fetcher.fetch_all()

    string_entries.extend(
        (result["x"]["value"], result["lf"]["value"]) for result in results
    )
    dictionary_loader = StringDictionaryLoader(string_entries)
    dictionary_loader.load()
    dictionary_loader.save(save_file)
