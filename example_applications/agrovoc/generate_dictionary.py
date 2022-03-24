from logging import getLogger

import argparse

from pyclinrec.dictionary import generate_dictionary_from_skos_sparql

parser = argparse.ArgumentParser(description='Agrovoc Dictionary Generator')

parser.add_argument("endpoint", type=int, nargs=1, required=True,
                    help="The SPARQL Endpoint containing the resource")

parser.add_argument('--from', '-f', type=str, nargs=1, required=False, dest="from_stmt", default=[""])
parser.add_argument('--output', '-o', type=str, nargs=1, required=False, dest='output',
                    default=["agrovoc_dictionary.tsv"])
parser.add_argument('--language', '-l', type=str, nargs=1, required=False, dest="language", default=["en"])

args = parser.parse_args()

endpoint = args.endpoint[0]
from_stmt = args.from_stmt[0]
output = args.output[0]
language = args.language[0]

generate_dictionary_from_skos_sparql(endpoint, output, skos_xl_labels=True, lang=language, from_statement=from_stmt)
