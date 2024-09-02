import redis
import torch
from torch.nn import Embedding

from torch.utils.data import Dataset
from tqdm import tqdm


class UMLSEmbeddings(Dataset):
    def __init__(self, embedding_file_descriptor):
        self.embedding_dict = {}
        embedding_tensor = []
        for idx, line in enumerate(embedding_file_descriptor):
            if isinstance(line, bytes):
                line = line.decode("utf8")
            line = line.strip().split(",")
            self.embedding_dict[line[0]] = idx
            # Convert the vectors on each line to float and add them to the embedding tensor list
            embedding_tensor.append([float(x) for x in line[1:]])
        self.embedding_tensor = torch.tensor(embedding_tensor)

    def __len__(self):
        return len(self.embedding_dict)

    def __getitem__(self, idx):
        if isinstance(idx, str):
            idx = self.embedding_dict[idx]
            return self.embedding_tensor[idx]
        elif isinstance(idx, int):
            return self.embedding_tensor[idx]


def load_umls_embeddings_to_redis(embedding_file_descriptor, redis_client):

    import redis
    from redis.commands.search.field import (
        TextField,
        VectorField,
    )
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
    from redis.commands.search.query import Query

    embedding_tensors = []
    cuis = []

    for idx, line in tqdm(
        enumerate(embedding_file_descriptor), desc="Processing UMLS embeddings"
    ):
        if isinstance(line, bytes):
            line = line.decode("utf8")
        line = line.strip().split(",")
        cuis.append(line[0])
        embedding_tensors.append([float(x) for x in line[1:]])

    pipeline = redis_client.pipeline()
    for cui, embedding in tqdm(
        zip(cuis, embedding_tensors),
        desc="Loading UMLS embeddings to Redis",
        total=len(embedding_tensors),
    ):
        global_key = f"umls:{cui}"
        pipeline.json().set(global_key, "$", {"cui": cui, "embedding": embedding})

    schema = (
        TextField("$.cui", no_stem=True, as_name="cui"),
        VectorField(
            "$.embedding",
            "FLAT",
            {
                "TYPE": "FLOAT32",
                "DIM": len(embedding_tensors[0]),
                "DISTANCE_METRIC": "COSINE",
            },
            as_name="vector",
        ),
    )
    print("Creating index...")
    definition = IndexDefinition(prefix=["umls:"], index_type=IndexType.JSON)
    redis_client.ft("idx:umls").create_index(fields=schema, definition=definition)


# class UMLSEmbeddingsVectorStore(Dataset):

#     def __init__(self, dataset: deeplake.dataset):
#         self.dataset = dataset

#     def __len__(self):
#         return len(self.store)

#     def __getitem__(self, idx):
#         if isinstance(idx, str):
#             return self.store.search(query="cui", value=idx)
#         elif isinstance(idx, int):
#             return self.embedding_tensor[idx]
