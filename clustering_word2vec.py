import pandas as pd
import numpy as np
import logging
import gensim.models
from gensim import utils
from gensim.corpora.wikicorpus import WikiCorpus


class SentencesCorpus:
    """An iterator that yields preprocessed sentences"""

    def __init__(self, corpus_filename: str):
        """
        Initializes the object and saves the given filename of the corpus (a .txt file with one sentence per line).
        :param corpus_filename:
        """
        self.corpus_filename = corpus_filename

    def __iter__(self):
        for line in open(self.corpus_filename):
            # assume there's one document per line, tokens separated by whitespace
            # tocens = utils.simple_preprocess(line)
            # if np.any(np.in1d(wordlist_include, tocens)) > 0:
            #     yield tocens
            yield utils.simple_preprocess(line)


class Word2VecModel:
    """A model for computing semantic similarity between words using a Word2Vec Gensim Model.
    """

    model: gensim.models.Word2Vec = None
    corpus: SentencesCorpus = None
    sim_matrix: pd.DataFrame = None

    @staticmethod
    def set_logging_info():
        """Sets the logging level to 'Info'."""
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    @staticmethod
    def set_logging_warning():
        """Sets the logging level to 'Warning' (to hide Info-Logging of model-fitting)."""
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)

    def load_model(self, filename: str):
        """
        Loads a gensim.models.Word2Vec Model from file
        :param filename: filename of the gensim model
        """
        self.model = gensim.models.Word2Vec.load(filename)

    def save_model(self, filename):
        """
        Saves a gensim.models.Word2Vec Model to disk.
        :param filename: filename which should be used for saving
        """
        self.model.save(filename)

    def load_wv(self, filename: str):
        """
        Loads Word Vectors from file.
        :param filename: filename of the word vectors
        """
        self.model = Word2VecModel()
        self.model.wv = gensim.models.KeyedVectors.load(filename)

    def save_wv(self, filename: str):
        """
        Save Word Vectors to file.
        :param filename: filename of the word vectors
        """
        self.model.wv.save(filename)

    @staticmethod
    def preprocess_wiki_corpus(filename_input: str, filename_output: str):
        """
        Preprocesses a wikipedia dump to a text-file with one preprocessed article per line.
        Set logging level to 'Info' to see progress output.
        :param filename_input: filename of the wikipedia dump (e.g. dewiki-latest-pages-articles.xml.bz2)
        :param filename_output: filename which should be used for saving the .txt file
        """
        logging.info("loading file " + filename_input + "...")
        wiki = WikiCorpus(filename_input, dictionary={})
        logging.info("Finished")
        logging.info("saving sentences to " + filename_output)

        output = open(filename_output, 'w')
        i = 0
        for text in wiki.get_texts():
            output.write(bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')
            i = i + 1
            if i % 10000 == 0:
                logging.info("Saved " + str(i) + " articles")

        output.close()
        logging.info("Finished Saved " + str(i) + " articles")

    def train_new_model(self, corpus_file: str, workers: int = 4, dimensions: int = 400, window: int = 5,
                        min_count: int = 5, sg: int = 1):
        """
        Trains a gensim word2vec Model on a given Text Corpus.
        :param corpus_file: The filename of the text corpus (a .txt file with one sentence per line)
        :param workers: # of cores used for computation (default: 4)
        :param dimensions: # of dimensions of the final vector word representation (default: 400)
        :param window: size of the window used for analyzing the context around each words (default: 5 words)
        :param min_count: How often a word needs to occur in the corpus to be analyzed (default: 5 times).
        :param sg: 1 for skip-gram; 0 for CBOW
        """
        self.corpus = SentencesCorpus(corpus_file)
        self.model = gensim.models.Word2Vec(sentences=self.corpus, workers=workers, vector_size=dimensions,
                                            window=window, min_count=min_count, sg=sg)

    def cosine_similarity(self, word1: str, word2: str) -> float:
        """
        Returns the cosine between the vectors corresponding to word 1 and word 2 as a measure of the similarity between
        both wordsword2vec.
        :param word1: one word
        :param word2: the other word
        :return: cosine between both vectors
        """
        word1 = word1.lower()
        word2 = word2.lower()
        if self.sim_matrix is not None and word1 in self.sim_matrix.columns and word2 in self.sim_matrix.index:
            return self.sim_matrix.loc[word1, word2]
        else:
            return self.model.wv.cosine_similarities(self.model.wv[word1], [self.model.wv[word2]])[0]

    def word_vector(self, word1: str) -> np.array:
        """
        Returns the vector of the given word
        :param word1: the word
        :return: the vector (numpy array)
        """
        return self.model.wv[word1]

    def word_exists(self, word: str) -> bool:
        return self.model.wv.__contains__(word.lower())

    def calculate_clusterids(self, intervals: pd.DataFrame, sim_threshold: float = 0.4,
                             clustering_type: str = "fixed_chain") -> np.array:
        """
        Calculates a np.array with IDs for each found cluster. If a word does not belong to a cluster, the value will
        be set to zero. Clusters are counted from 1 to cluster_max. Clusters are defined as a chain of words where
        all neighbors have a minimum semantic relatedness of sim_threshold.
        :param intervals: pd.DataFrame with column 'word'
        :param sim_threshold: threshold used for defining clusters of semantic related words. If any dynamic clustering
        algorithm is used, this value will we ignored
        :param clustering_type: used clustering-mechanism (possible values are: 'fixed_chain', 'fixed_cluster',
        'dynamic_chain', 'dynamic_cluster') See Linz et. al 2017 for details.
        :return: the given pd.DataFrame with 1 additional column: cluster (indicating the cluster ID)
        """
        cluster_ids = [np.NAN for _ in intervals.index]
        cluster_ids[0] = 0
        cluster_id = 0

        intervals.reset_index(drop=True, inplace=True)

        # check if clustertype is valid
        if clustering_type != "dynamic_chain" and clustering_type != "dynamic_cluster" and \
                clustering_type != "fixed_chain" and clustering_type != "fixed_cluster":
            raise Exception("unknown clustering type " + clustering_type)

        # dynamic threshold calculation
        if clustering_type == "dynamic_chain" or clustering_type == "dynamic_cluster":
            sim_threshold = self.calculate_dynamic_threshold(intervals)
            # print("using dynamic similarity threshold " + str(sim_threshold) + " for " + clustering_type)
        # else:
            # print("using fixed similarity threshold " + str(sim_threshold) + " for " + clustering_type)

        for i in range(1, intervals.shape[0]):

            if clustering_type == "dynamic_chain" or clustering_type == "fixed_chain":
                try:
                    is_cluster = self.cosine_similarity(intervals.loc[i, "word"],
                                                        intervals.loc[i - 1, "word"]) > sim_threshold
                except KeyError:
                    is_cluster = False
                    print("WARNING: word pair not found: " + intervals.loc[i, "word"] + " - " + intervals.loc[i - 1, "word"])
            else:
                raise NotImplementedError("clusterin_type " + clustering_type + " not implemented yet!")

            if not is_cluster:
                cluster_id += 1

            cluster_ids[i] = cluster_id

        return cluster_ids

    def calculate_dynamic_threshold(self, intervals: pd.DataFrame) -> float:
        """
        Calculates the individual threshold for word2vec cosine similarity for a given word list.
        See Linz et. al 2017 for details.
        :param intervals: pd.DataFrame with column 'word'
        :return: individual calculated threshold for word2vec cosine similarity calculation
        (=mean of all word pairs in intervals["word"])
        """
        pair_count = 0
        sim_all = 0

        intervals.reset_index(drop=True, inplace=True)

        for i in range(0, intervals.shape[0] - 1):
            for j in range(0, intervals.shape[0] - 1):
                if i == j:
                    continue

                word1 = intervals.loc[i, "word"]
                word2 = intervals.loc[i + 1, "word"]

                try:
                    sim = self.cosine_similarity(word1, word2)
                    sim_all += sim
                    pair_count += 1
                except KeyError:
                    pass
        sim_all /= pair_count
        sim_threshold = sim_all
        return sim_threshold

    def create_similarity_matrix(self, wordlist: list):
        """
        Creates a cosine similarity matrix between all words given in the wordlist. The matrix is stored in the variable
        sim_matrix.
        :param wordlist: list of words for which the similarity matrix should be created
        """
        wordlist = list(map(lambda s: s.lower(), wordlist))
        wordlist = pd.Series(wordlist).unique()
        self.sim_matrix = pd.DataFrame(np.nan, columns=wordlist, index=wordlist)
        for word1 in wordlist:
            for word2 in wordlist:
                try:
                    self.sim_matrix.loc[word1, word2] = self.model.wv.cosine_similarities(self.model.wv[word1],
                                                                                          [self.model.wv[word2]])[0]
                except KeyError:
                    pass

    def calculate_mean_seqrel_total(self, wordlist: list) -> float:
        """
        Calculates the mean sequential relatedness of all sequential word pairs
        :param wordlist: list of words
        :return: mean of sequential relatedness
        """
        wordlist = list(map(lambda s: s.lower(), wordlist))

        wordcount = 0
        sim_total = 0

        for i in range(0, len(wordlist) - 1):
            word1 = wordlist[i]
            word2 = wordlist[i + 1]
            try:
                sim = self.cosine_similarity(word1, word2)
                wordcount += 1
                if not np.isnan(sim):
                    sim_total += sim
                # print(word1 + " - " + word2 + ": " + str(sim))
            except KeyError:
                # print(word1 + " - " + word2 + ": not found")
                pass

        if wordcount == 0:
            return np.NAN

        return sim_total / wordcount

    def calculate_mean_seqrel_percluster(self, intervals: pd.DataFrame) -> float:
        """
        Calculates the mean sequential relatedness of each cluster and returns the mean of these mean values.
        :param intervals: pd.DataFrame with column 'word' and 'cluster'
        :return: mean of sequential relatedness per cluster
        """
        # calculate mean seqrel for each cluster using the calculate_mean_seqrel_total function
        clusters = intervals.groupby("cluster")["word"].apply(lambda s: self.calculate_mean_seqrel_total(s))
        clusters = clusters[1:]  # remove the cluster '0' which is no cluster
        return clusters.mean()

    def calculate_mean_cumrel_total(self, wordlist: list) -> float:
        """
        Calculates the mean cumulative relatedness of all word pairs in the wordlist
        :param wordlist: list of words
        :return: mean of cumulative relatedness
        """
        wordlist = list(map(lambda s: s.lower(), wordlist))

        wordcount = 0
        sim_total = 0

        for word1 in wordlist:
            for word2 in wordlist:
                if word1 == word2:
                    continue

                try:
                    sim = self.cosine_similarity(word1, word2)
                    wordcount += 1
                    if not np.isnan(sim):
                        sim_total += sim
                    # print(word1 + " - " + word2 + ": " + str(sim))
                except KeyError:
                    # print(word1 + " - " + word2 + ": not found")
                    pass

        if wordcount == 0:
            return np.NAN
        return sim_total / wordcount

    def calculate_mean_cumrel_percluster(self, intervals: pd.DataFrame) -> float:
        """
        Calculates the mean cumulative relatedness of each cluster and returns the mean of these mean values.
        :param intervals: pd.DataFrame with columns 'word' and 'cluster'
        :return: mean of cumulative relatedness per cluster
        """
        # calculate mean seqrel for each cluster using the calculate_mean_seqrel_total function
        clusters = intervals.groupby("cluster")["word"].apply(lambda s: self.calculate_mean_cumrel_total(s))
        clusters = clusters[1:]  # remove the cluster '0' which is no cluster
        return clusters.mean()
