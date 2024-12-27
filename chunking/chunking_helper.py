import csv
import gzip
import re

from unidecode import unidecode

from abc import ABC
from dataclasses import dataclass
from typing import List, Optional
from ranking.rank_bm25 import BM25Okapi
from llm.llms import Role


@dataclass
class Chunk:
    size: int
    text: str
    role: Optional[Role] = None
    similarity: Optional[float] = None
    include_in_request: Optional[bool] = None
    # flag to identify if there is only functioncall in the assistant message or
    # functioncall exists interleaving with response
    is_intext_functioncall: Optional[bool] = None


class SimilarityCalculator(ABC):
    def __init__(self):
        self.threshold = 0.0

    def calculate_similarity(self, query: str, target_chunks: List[Chunk], explain=False):
        pass


def _load_word_frequency(filepath):
    ret = {}  # word -> freq
    with gzip.open(filepath, "rt") as file:
        csvreader = csv.reader(file, delimiter="\t", escapechar=None, doublequote=None, quotechar=None)
        for row in csvreader:
            if len(row) != 2:
                continue
            try:
                freq = int(row[0])
                word = row[1]
                # tokenizer, normalize and combine counts
                for token in tokenize(word):
                    ret[token] = ret.get(token, 0) + freq
            except Exception as e:
                print(e)
                pass
    return ret


stop_words = set()
with open("chunking/stop_words.txt") as file:
    stop_words.update([line.rstrip() for line in file])


def is_stop_word(word):
    if word in stop_words:
        return True
    if len(word) <= 1:
        return True
    return False


non_alphanumeric = re.compile(r'[^a-z0-9]')


def tokenize(doc):
    doc = unidecode(doc)
    doc = doc.lower()
    words = doc.split()
    normalized_words = []
    for word in words:
        # TODO: rewrite this without using regex for better performance
        normalized_word = non_alphanumeric.sub(" ", word)
        if word != normalized_word:
            normalized_words.extend(normalized_word.split())
    words.extend(normalized_words)
    words = [word for word in words if not is_stop_word(word)]
    return words


def remove_urls(text):
    text = text.lower()
    ret = []
    for chunk in text.split():
        if chunk.startswith("http://") or chunk.startswith("https://"):
            continue
        ret.append(chunk)
    return " ".join(ret)


def normalize_query_for_truncation(query):
    query = query.lower()
    query = remove_urls(query)
    return query


class BM25Calculator(SimilarityCalculator):
    _external_word_freq = _load_word_frequency("ranking/wp_1gram_top1m.txt.gz")
    _bm25 = BM25Okapi(external_word_freq=_external_word_freq)

    def __init__(self):
        super().__init__()
        self.threshold = 0.19

    def calculate_similarity(self, query: str, target_chunks: List[Chunk], explain=False):
        corpus = [chunk.text for chunk in target_chunks]
        tokenized_corpus = [tokenize(doc) for doc in corpus]
        query = normalize_query_for_truncation(query)
        query = tokenize(query)
        explain_dict = {} if explain else None
        scores = self._bm25.get_scores(query, tokenized_corpus, max_term_frequency=10,
                                       recalculate_idf=False, explain=explain_dict)
        scores = self.normalize(scores)
        for chunk, score in zip(target_chunks, scores):
            chunk.similarity = score

        if explain:
            for idx, chunk in enumerate(target_chunks):
                chunk.explain_similarity = explain_dict.get(idx)

    @staticmethod
    def normalize(scores):
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [0] * len(scores)
        return [(score - min_score) / (max_score - min_score) for score in scores]


def get_similarity_calculator(similarity_type: str = "bm25") -> SimilarityCalculator:
    return BM25Calculator()


if __name__ == '__main__':
    pass


