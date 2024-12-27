#!/usr/bin/env python

import math
import numpy as np

"""
Initial code from: https://github.com/dorianbrown/rank_bm25
We've added support for external frequency counts.
"""

class BM25:
    def __init__(self, external_word_freq: dict[str,int]=None):
        self.idf = {}
        self.highest_idf = 0
        self._calc_idf(word_freq=external_word_freq)

    def _calc_idf(self, word_freq: dict[str,int]):
        raise NotImplementedError()

    def get_scores(self, query, corpus, max_term_frequency=0, recalculate_idf=False, explain=False):
        raise NotImplementedError()

class BM25Okapi(BM25):
    def __init__(self,
                 k1=1.5,         # controls how document frequency influences score. k1=0 means zero influence.
                 b=0.1,          # controls how document length influences score. b=0 means zero influence.
                 epsilon=0.0,    # assigns an idf value for words that have negative idf
                 external_word_freq=None):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(external_word_freq=external_word_freq)

    def _calc_idf(self, word_freq: dict[str,int]):
        """
        Calculates idf based on frequency dict
        """
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        fake_corpus_size = max(word_freq.values()) * 2
        for word, freq in word_freq.items():
            idf = math.log(fake_corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps
        self.highest_idf = max(self.idf.values())

    def get_scores(self, query, corpus, max_term_frequency=0, recalculate_idf=False, explain: dict = None):
        """
        The ATIRE BM25 variant uses an idf function which uses a log(idf) score. To prevent negative idf scores,
        this algorithm also adds a floor to the idf value of epsilon.
        See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info
        """
        corpus_size = 0
        doc_freqs = []
        doc_len = []

        nd = {}  # word -> number of documents with word
        num_doc = 0
        for document in corpus:
            doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                if max_term_frequency > 0:
                    frequencies[word] = min(max_term_frequency, frequencies[word] + 1)
                else:
                    frequencies[word] += 1
            doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                nd[word] = nd.get(word, 0) + 1
            corpus_size += 1

        avg_doc_len = num_doc / corpus_size

        score = np.zeros(corpus_size)
        doc_len = np.array(doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in doc_freqs])
            score += self.get_idf(q) * (q_freq * (self.k1 + 1) /
                                        (q_freq + self.k1 * (1 - self.b + self.b * doc_len / avg_doc_len)))
            if explain is not None:
                for idx, doc in enumerate(doc_freqs):
                    if doc.get(q):
                        if not explain.get(idx):
                            explain[idx] = {}
                        explain[idx][q] = self.get_idf(q) * (q_freq[idx] * (self.k1 + 1) /
                                        (q_freq[idx] + self.k1 * (1 - self.b + self.b * doc_len[idx] / avg_doc_len)))

        return score

    def get_idf(self, q):
        return self.idf.get(q, self.highest_idf)