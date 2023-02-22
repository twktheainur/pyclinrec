import pickle

from pathlib import Path

endpoint = "https://data-issa.cirad.fr/sparql"  # args.endpoint[0]
from_stmt = "http://agrovoc.fao.org/graph"  # args.from_stmt[0]
language = "fr"  # args.language[0]
output = f"agrovoc-{language}.tsv"  # args.output[0]

import os

from pyclinrec.dictionary import generate_dictionary_from_skos_sparql


class AgrovocDictionaryGenerator:
    def __init__(self,
                 endpoint="https://data-issa.cirad.fr/sparql",
                 graph="http://agrovoc.fao.org/graph",
                 language="en",
                 output_dir='.'):
        self.endpoint = endpoint
        self.graph = graph
        self.language = language

        self.output = os.path.join(output_dir, f"agrovoc-{language}.tsv")

        if not os.path.exists(self.output):
            Path(output_dir).mkdir(exist_ok=True)
            # generate dict tsv file
            print('generating dictionary..')
            generate_dictionary_from_skos_sparql(endpoint, self.output,
                                                 skos_xl_labels=True,
                                                 lang=language,
                                                 from_statement=graph)


dict_gen_en = AgrovocDictionaryGenerator(output_dir='./vocab', language="en")
dict_gen_fr = AgrovocDictionaryGenerator(output_dir='./vocab', language="fr")

from pyclinrec.dictionary import MgrepDictionaryLoader
from pyclinrec.recognizer import IntersStemConceptRecognizer

from pyclinrec import __path__ as pyclinrec_path


class AgrovocAnnotator:
    def __init__(self, dictionary_file, language="en"):
        dictionary_loader = MgrepDictionaryLoader(dictionary_file)

        self.concept_recognizer = IntersStemConceptRecognizer(dictionary_loader,
                                                              os.path.join(pyclinrec_path[0],
                                                                           f"stopwords{language}.txt"),
                                                              os.path.join(pyclinrec_path[0],
                                                                           f"termination_terms{language}.txt"))

        self.concept_recognizer.initialize()

    def annotate(self, text):
        return self.concept_recognizer.annotate(text)


if __name__ == "__main__":
    pkl_file = 'AgrovocAnnotator_en.pkl'

    if not os.path.exists(pkl_file):
        annotator_en = AgrovocAnnotator("./vocab/agrovoc-en.tsv", language="en")

        # serialise to save time on initialization
        with open(pkl_file, 'wb') as f:
            pickle.dump(annotator_en, f)
    else:
        # deserialize
        with open(pkl_file, 'rb') as f:
            annotator_en = pickle.load(f)

    len(annotator_en.concept_recognizer.concept_index)

    pkl_file = 'AgrovocAnnotator_fr.pkl'

    if not os.path.exists(pkl_file):
        annotator_fr = AgrovocAnnotator("./vocab/agrovoc-fr.tsv", language="fr")

        # serialise to save time on initialization
        with open(pkl_file, 'wb') as f:
            pickle.dump(annotator_fr, f)
    else:
        # deserialize
        with open(pkl_file, 'rb') as f:
            annotator_fr = pickle.load(f)
    len(annotator_fr.concept_recognizer.concept_index)

    text = "Agwergsd. Plant-plant polination is possible"

    annotations = annotator_en.annotate(text)

    print(len(annotations[2]))

    nul = [print(a) for a in sorted(list(annotations[2]), key=lambda x: x.concept_id)]

    text = """
    Les récents progrès des technologies à haut débit ont entraîné une explosion de la quantité de données dans le domaine agronomique. Il est urgent d'intégrer efficacement des informations complémentaires pour comprendre le système biologique dans sa globalité. Nous avons développé AgroLD, une base de connaissances qui exploite la technologie du Web sémantique et des ontologies du domaine biologique pertinentes, pour intégrer les informations sur les espèces végétales et faciliter ainsi la formulation de nouvelles hypothèses scientifiques. Nous présentons des résultats sur le processus d'intégration et sur la plateforme visualisation des données, qui était initialement axé sur la génomique, la protéomique et la phénomique.
    """
    annotations = annotator_fr.annotate(text)

    print(len(annotations[2]))

    nul = [print(a) for a in sorted(list(annotations[2]), key=lambda x: x.start)]
