import os
import logging
import multiprocessing
import argparse
import numpy as np
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
            yield utils.simple_preprocess(line)


# Command Line Argument Parsing - Initialization
parser = argparse.ArgumentParser(prog="Word2Vec Model Trainer for Wikipedia Corpus",
                                 description="This script trains a word2vec model based on a Wikipedia text corpus ("
                                             "prodived as .xml.bz2-file) --- The Wikipedia corpus can be obtained "
                                             "here: "
                                             "https://dumps.wikimedia.org/backup-index.html --- Example: python "
                                             "modeltraining_word2vec.py -i dewiki-20221020-pages-articles.xml.bz2 -p "
                                             "dewiki-20221020-pages-articles.txt -o "
                                             "dewiki-20221020-pages-articles.kv")
parser.add_argument("-i", "--Input", help="Wikipedia corpus used as input file (.xml.bz2-file)", metavar="input_file",
                    required=True)
parser.add_argument("-p", "--Preprocessed", help="file of preprocessed Wikipedia corpus (.txt). If this file already "
                                                 "exists, it will be used for model training. Else it will be created"
                                                 " from the input file", metavar="preprocessed_file", required=True)
parser.add_argument("-o", "--Output", help="output file for trained model (.kv-file)", metavar="output_file",
                    required=True)
parser.add_argument("-w", "--Workers", help="number of workers (default=number of CPU cores - 1)",
                    metavar="number_of_workers", type=int)
parser.add_argument("-d", "--Dimensions", help="number of dimensions of the word2vec model (default=500)",
                    metavar="number_of_dimensions", default=500, type=int)
parser.add_argument("-s", "--Size", help="size of window used for word2vec model training (default=10)",
                    metavar="window_size", default=10, type=int)
parser.add_argument("-m", "--Min", help="minimum word count used for word2vec model training (default=5)",
                    metavar="minimum_word_count", default=5, type=int)
parser.add_argument("-a", "--Algorithm", help="algorithm used for word2vec model training (default=1; 1 for "
                                              "skip-gram; 0 for CBOW)", metavar="algorithm_type", default=1, type=int,
                    choices=[0, 1])

args = parser.parse_args()

# print the arguments used
print("Word2Vec Model Trainer for Wikipedia Corpus")
print("")
print("The following files will be used: ")
print("Input file: " + args.Input)
print("Preprocessing file: " + args.Preprocessed)
print("Output file: " + args.Output)
print("")
print("The following parameters will be used:")
print("Dimensions: " + str(args.Dimensions))
print("Window Size: " + str(args.Size))
print("Minimum Word Count: " + str(args.Size))
print("Algorithm (1=skip-gram, 0=CBOW): " + str(args.Algorithm))
print("")
# use cpu core number - 1 if no number of workers is provided
if args.Workers is None:
    workers = np.max([multiprocessing.cpu_count() - 1, 1])
else:
    workers = args.Workers
print("Training will be performed using " + str(workers) + " workers")

# enable logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# text preprocessing (converting to simple text file) if not already performed
if os.path.exists(args.Preprocessed):
    print("preprocessing skipped because output file " + args.Preprocessed + " already exists")
else:
    print("Starting preprocessing...")
    wiki = WikiCorpus(args.Input, dictionary={})
    output = open(args.Preprocessed, 'w')
    for text in wiki.get_texts():
        output.write(bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')
    output.close()
    print("Finished preprocessing")

# model training if not already performed
if os.path.exists(args.Output):
    print("model training skipped because output file " + args.Output + " already exists")
else:
    print("Starting model training...")

    # Sentence Corpus is used for simple preprocessing of the Wikipedia File
    corpus = SentencesCorpus(args.Preprocessed)

    # start the model training using the hyperparameters stated above
    model = gensim.models.Word2Vec(sentences=corpus, workers=workers, vector_size=args.Dimensions,
                                   window=args.Size, min_count=args.Min, sg=args.Algorithm)

    # save the model without internal weights
    model.wv.save(args.Output)
    del model.wv
    del model
    print("Finished training.")
