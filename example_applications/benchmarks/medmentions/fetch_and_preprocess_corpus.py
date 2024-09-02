import hashlib
import json
import pickle
import redis
import requests

from tqdm import tqdm

from medmentions.medmentions import (
    Document,
    Mention,
)


redis_client = redis.StrictRedis(host="localhost", port=6379, db=0)


def get_with_cache(url, headers=None):
    # Create a unique key for the URL
    url_hash = hashlib.md5(url.encode()).hexdigest()
    if cached_response := redis_client.get(url_hash):
        return json.loads(cached_response.decode("utf-8"))
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        redis_client.set(url_hash, response.text)
    return json.loads(response.text)


class InvalidPubtatorFormatError(Exception):
    def __init__(self, message):
        super().__init__(message)


def parse_pubtator(corpus_file_descriptor, limit=None):
    from tqdm import tqdm

    lines = corpus_file_descriptor.readlines()
    docs = []
    concept_set = set()
    doc_mentions = []
    current_id = None
    current_title = None
    current_abstract = None
    abstract_passed = False
    document_count = 0
    for line in tqdm(lines):
        # Transform b string to string
        line = line.decode("utf-8")
        if "|t|" in line:
            title_data = line.split("|t|")
            if current_id is None or current_id != title_data[0]:
                document_count += 1
                if limit is not None and document_count > limit:
                    break
                if len(doc_mentions) > 0:
                    docs.append(
                        Document(
                            current_id, current_title + current_abstract, doc_mentions
                        )
                    )
                    doc_mentions = []
                    abstract_passed = False
                current_id = title_data[0]
                current_title = title_data[1]
        elif not abstract_passed:
            if "|a|" not in line:
                raise InvalidPubtatorFormatError("Invalid syntax, expected abstract")
            abstract_data = line.split("|a|")
            current_abstract = abstract_data[1]
            abstract_passed = True
        elif line != "\n":
            mention_data = line.split("\t")
            start = int(mention_data[1])
            end = int(mention_data[2])
            mention = Mention(
                doc_id=mention_data[0],
                sem_type=mention_data[4],
                linked_class=mention_data[5].strip(),
                start=start,
                end=end,
                text=mention_data[3],
            )
            if mention.linked_class not in concept_set:
                concept_set.add(mention.linked_class)
            doc_mentions.append(mention)

    return docs, concept_set


def get_all_pages(uri):

    response = get_with_cache(uri)

    # print(f"Retrieving all pages of  {uri}...")
    if response["pageCount"] > 1:
        if not isinstance(response["result"], list):
            response["result"] = [response["result"]]
        for page_number in range(2, int(response["pageCount"])):
            page_response = get_with_cache(f"{uri}&pageNumber={page_number}")
            if not isinstance(page_response["result"], list):
                response["result"].extend([page_response["result"]])
            else:
                response["result"].extend(page_response["result"])
    return response


def fetch_CUI_data_from_UMLS(
    cui, umls_version="2017AA", api_key=None, include_relations=False
):
    if api_key is not None:
        url = "https://uts-ws.nlm.nih.gov/rest"

    param_string = f"?apiKey={api_key}"
    query = f"{url}/content/{umls_version}/CUI/{cui}{param_string}"
    response = get_with_cache(query)

    definitions_uri = response["result"]["definitions"]
    final_dict = {"definitions": []}
    if definitions_uri != "NONE":
        definitions = get_all_pages(definitions_uri + param_string)
        for definition in definitions["result"]:
            source = definition["rootSource"]
            text = definition["value"]
            final_dict["definitions"].append({"source": source, "text": text})

    atoms_uri = response["result"]["atoms"]
    atoms = get_all_pages(atoms_uri + param_string)
    final_dict["labels"] = []
    if atoms_uri != "NONE":
        for atom in atoms["result"]:
            code = atom["code"]
            concept = atom["concept"]
            name = atom["name"]
            term_type = atom["termType"]
            final_dict["labels"].append(
                {"code": code, "concept": concept, "name": name, "termType": term_type}
            )

    if include_relations:
        relations_uri = response["result"]["relations"]
        if relations_uri != "NONE":
            relations = get_all_pages(relations_uri + param_string)

            response["result"]["relations"] = []
            for relation in relations["result"]:
                print(relation)
                relation_name = relation["additionalRelationLabel"]
                relation_type = relation["relationLabel"]
                relation_origin = relation["rootSource"]
                rid = relation["ui"]
                response["result"]["relations"].append(
                    {
                        "name": relation_name,
                        "type": relation_type,
                        "origin": relation_origin,
                        "id": rid,
                    }
                )

            final_dict["relations"] = response["result"]["relations"]

    return final_dict


#

# Step 1: Download the zip file
url = "https://github.com/chanzuckerberg/MedMentions/raw/master/st21pv/data/corpus_pubtator.txt.gz"
response = requests.get(url, stream=True)

# Sizes in bytes.
total_size = int(response.headers.get("content-length", 0))
block_size = 1024

with tqdm(
    total=total_size,
    unit="B",
    unit_scale=True,
    desc="Downloading corpus corpus_pubtator.txt.gz",
) as progress_bar:
    with open("data/corpus_pubtator.txt.gz", "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

if total_size != 0 and progress_bar.n != total_size:
    raise RuntimeError("Could not download file")

# Step 2: Extract the file from the gz archive and parse its contents

import gzip

with gzip.open("data/corpus_pubtator.txt.gz", "r") as f:
    from tqdm import tqdm

    medmentions, concepts = parse_pubtator(f, limit=40)

    concepts = {
        concept.split(":")[1]: fetch_CUI_data_from_UMLS(
            concept.split(":")[1],
            umls_version="2017AA",
            api_key="603302b8-6645-40a6-a944-39319c399451",
        )
        for concept in tqdm(concepts)
    }

    pickle.dump(medmentions, open("data/medmentions.pkl", "wb"))
    pickle.dump(concepts, open("data/concepts.pkl", "wb"))

    with open("data/medmentions.json", "w") as mmf:
        mmf.write("[\n")
        for doc in medmentions:
            mmf.write(doc.toJSON() + ",\n")
        mmf.write("]\n")

    json.dump(concepts, open("data/concepts.json", "w"))
