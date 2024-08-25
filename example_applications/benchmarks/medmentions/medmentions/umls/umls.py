import torch
from torch.nn import Embedding

from torch.utils.data import Dataset


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
