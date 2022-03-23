from logging import getLogger

from rdflib import Graph, Namespace

from pyclinrec.dictionary import StringDictionaryLoader
from pyclinrec.recognizer import IntersStemConceptRecognizer

from sparql_utils import SparQLOffsetFetcher

logger = getLogger("Agrovoc")


def generate_brat_normalization_database(string_entries, target_file="brat_norm_db.txt", remove_uris=True):
    concept_dictionary = {}
    for item in string_entries:
        if remove_uris:
            key = item[0].split("/")[-1].split("_")[1]
        else:
            key = item[0]
        if key not in concept_dictionary:
            concept_dictionary[key] = []
        concept_dictionary[key].append(item[1])
    with open(target_file, "w") as target_handler:
        for key in concept_dictionary.keys():
            values = concept_dictionary[key]
            entry = f"{key}"
            for value in values:
                entry += f"\tname:Name:{value}"
            entry += '\n'
            target_handler.write(entry)


def generate_dictionary_from_skos_file(self, graph: Graph = None, thesaurus_path=None,
                                       save_file="agrovoc_dictionary.tsv",
                                       skos_xl_labels=True,
                                       lang="fr"):
    if graph is None:
        self.graph = Graph()
    else:
        self.graph = graph
    logger.info("Loading thesaurus... [{}]".format(thesaurus_path))

    self.graph.load(thesaurus_path)

    string_entries = []

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

    for result in pref_labels:
        string_entries.append((str(result[0]), str(result[1])))

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

    for result in alt_labels:
        string_entries.append((str(result[0]), str(result[1])))
    generate_brat_normalization_database(string_entries, remove_uris=True)
    dictionary_loader = StringDictionaryLoader(string_entries)
    dictionary_loader.load()
    dictionary_loader.save(save_file)


def generate_dictionary_from_skos_sparql(self, endpoint=None,
                                         save_file="agrovoc_dictionary.tsv",
                                         skos_xl_labels=True,
                                         lang="fr"):
    sparql = SPARQLWrapper(endpoint)

    string_entries = []
    if skos_xl_labels:
        fetcher = SparQLOffsetFetcher(sparql, 9000,
                                      where_body=f"?x a skos:Concept;\nskosxl:prefLabel ?l.\n ?l skosxl:literalForm ?lf.\nFILTER(lang(?lf)='{lang}'",
                                      select_columns="?x ?lf",
                                      prefixes="prefix skos: <http://www.w3.org/2004/02/skos/core#>\n"
                                               "prefix skosxl: <http://www.w3.org/2008/05/skos-xl#>")
    else:
        fetcher = SparQLOffsetFetcher(sparql, 9000,
                                      where_body=f"?x a skos:Concept;\nskos:prefLabel ?lf.\nFILTER(lang(?lf)='{lang}'",
                                      select_columns="?x ?lf",
                                      prefixes="prefix skos: <http://www.w3.org/2004/02/skos/core#>")
    results = fetcher.fetch_all()
    for result in results:
        string_entries.append((result["x"]["value"], result["lf"]["value"]))
    if skos_xl_labels:
        fetcher = SparQLOffsetFetcher(sparql, 9000,
                                      where_body=f"?x a skos:Concept;\nskosxl:altLabel ?l.\n ?l skosxl:literalForm ?lf.\nFILTER(lang(?lf)='{lang}'",
                                      select_columns="?x ?lf",
                                      prefixes="prefix skos: <http://www.w3.org/2004/02/skos/core#>\n"
                                               "prefix skosxl: <http://www.w3.org/2008/05/skos-xl#>")
    else:
        fetcher = SparQLOffsetFetcher(sparql, 9000,
                                      where_body=f"?x a skos:Concept;\nskos:altLabel ?lf.\nFILTER(lang(?lf)='{lang}'",
                                      select_columns="?x ?lf",
                                      prefixes="prefix skos: <http://www.w3.org/2004/02/skos/core#>")
    results = fetcher.fetch_all()

    for result in results:
        string_entries.append((result["x"]["value"], result["lf"]["value"]))

    dictionary_loader = StringDictionaryLoader(string_entries)
    dictionary_loader.load()
