import sys, string
import json, pickle
import numpy as np
from collections import defaultdict
from data_utils import *
import gensim

class DataProcessor(object):

    def __init__(self, data_path):
        self.train_data_path = data_path + "train.clean"
        self.dev_data_path = data_path + "dev.clean"
        self.test_data_path = data_path + "test.clean"
        self.vocab_path = data_path + "vocab.txt"
        self.char_path = data_path + "vocab_char.txt"
        self.config_path = data_path + "config.txt"
        self.tags = {"O":0, "PER":1, "MISC":2, "ORG":3, "LOC":4}
        self.prepare()
        
    
    def prepare(self):
        self.config = json.loads(open(self.config_path).read())
        # load vocabulary
        sys.stderr.write("loading vocabulary...")
        with open(self.vocab_path, "r") as fi:
            self.vocab = {x.split()[0]: int(x.split()[1]) for x in fi}
            self.id2vocab = {v: k for k,v in self.vocab.items()}
        sys.stderr.write("done.\n")

        # load character vocabulary
        sys.stderr.write("loading characters...")
        with open(self.char_path, "r") as fi:
            self.char = {x.split()[0]: int(x.split()[1]) for x in fi}
            self.id2char = {v: k for k,v in self.char.items()}
        sys.stderr.write("done.\n")

        # load train/dev/test data
        sys.stderr.write("loading dataset...")
        self.train_data = self._load_dataset(self.train_data_path)
        self.dev_data = self._load_dataset(self.dev_data_path)
        self.test_data = self._load_dataset(self.test_data_path)
        sys.stderr.write("done.\n")
        
        # load word vector
        sys.stderr.write("loading wordvector...")
        self.vocab_vec = self._load_vector()
        sys.stderr.write("done.\n")

    def _load_dataset(self, path):
        dataset = []
        with open(path, "r") as input_data:
            for line in input_data:
                tokens = [x for x in json.loads(line)]
                token_ids = [self.vocab[x["surface"]] if x["surface"] \
                             in self.vocab else self.vocab["<UNK>"] for x in tokens]
                tokens_addition = [check_additon_word(x["raw"]) for x in tokens]
                sent_len = len(token_ids)
                token_ids += [0 for _ in range(self.config["max_sent_len"] - sent_len)]
                tokens_addition += [0 for _ in range(self.config["max_sent_len"] - sent_len)]
                targets = [self.tags[clear_target(x["target"])] for x in tokens]
                chars = [clear_char(token["raw"], self.config["max_word_len"], self.char) for token in tokens]
                chars += [[0 for _ in range(self.config["max_word_len"])] \
                          for _ in range(self.config["max_sent_len"] - sent_len)]
                chars_addtion = [clear_char_addition(x, self.config["max_word_len"]) for x in tokens]
                chars_addtion += [[0 for _ in range(self.config["max_word_len"])] \
                          for _ in range(self.config["max_sent_len"] - sent_len)]
                dataset.append((token_ids, sent_len, tokens_addition, chars, chars_addtion,  targets))
        return dataset

    def _load_vector(self):    
        sys.stderr.write("Loading Glove embeddings...\n")
        glove_path = "glove.6B.50d.txt"
        glove_vectors, glove_dict = load_glove_vectors(glove_path, vocab=set(self.vocab.keys()))
        vocab_vec = build_initial_embedding_matrix(self.vocab, glove_dict, glove_vectors,)
        return vocab_vec


if __name__ == "__main__":
    data_processor = DataProcessor("work/")
    sys.stderr.write("saving dataset ....")
    pickle.dump(data_processor.train_data, open("train.data", "wb"))
    pickle.dump(data_processor.test_data, open("test.data", "wb"))
    pickle.dump(data_processor.dev_data, open("dev.data", "wb"))
    pickle.dump(data_processor.vocab_vec, open("vocab.vec", "wb"))
    sys.stderr.write("done.\n")
