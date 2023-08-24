import math as math
from typing import List, Tuple

from nltk import TreebankWordTokenizer
from nltk.corpus import stopwords

tokenizer = TreebankWordTokenizer()

_stop_words = stopwords.words('english')


def compute_hard_overlap(collection_a: List[str], collection_b: List[str]):
    overlap_count = 0
    for item in collection_a:
        item_a = item
        for item_ in collection_b:
            item_b = item_
            if item_a == item_b:
                overlap_count += 1
    return overlap_count


def tverski_ratio(alpha: float, beta: float, gamma: float, overlap_count: float, difference_a: float,
                  difference_b: float):
    contrast = tverski_contrast(alpha, beta, gamma, overlap_count, difference_a, difference_b)
    return 0 if contrast == 0 else alpha * overlap_count / contrast


def tverski_contrast(alpha: float, beta: float, gamma: float, overlap_count: float, difference_a: float,
                     difference_b: float):
    return alpha * overlap_count - beta * difference_a - gamma * difference_b


def jaccard_count(overlap_count: float, union_count: float):
    return 0 if union_count == 0 else overlap_count / union_count


def jaccard(collection_a, collection_b):
    overlap = compute_hard_overlap(collection_a, collection_b)
    return jaccard_count(overlap, len(collection_a) + len(collection_b))


def geometric_mean_aggregation(weighted_values: List[Tuple[float, float]]):
    length = len(weighted_values)
    overall_product = 1
    for (v, w) in weighted_values:
        if v is not None:
            v = max(v, 0.00001)
            overall_product *= math.pow(v, w)
    return math.pow(overall_product, 1.0 / float(length))


def arithmetic_mean_aggregation(weighted_values: List[Tuple[float, float]]):
    length = len(weighted_values)
    overall_sum = 0.0
    for (v, w) in weighted_values:
        overall_sum += v * w

    return overall_sum / float(length)
