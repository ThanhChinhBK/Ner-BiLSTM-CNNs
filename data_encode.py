import sys, string
import json
import numpy as np
from collections import defaultdict
from data_utils import *

class DataProcessor(object):

    def __init__(self, data_path):
        self.train_data_path = data_path + "train.clean"
        self.dev_data_path = data_path + "dev.clean"
        self.test_data_path = data_path + "test.clean"
        self.vocab_path = data_path + "vocab.txt"
        self.char_path = data_path + "vocab_char.txt"
        self.tags = {"O":0, "PER":1, "MISC":2, "ORG":3, "LOC":4}
        self.prepare()
        
    
    def prepare(self):
        # load vocabulary
        sys.stderr.write("loading vocabulary...")
        with open(self.vocab_path, "r") as fi:
            self.vocab = {x.split()[0]: int(x.split()[1]) for x in fi}
            self.id2vocab = {v: k for k,v in self.vocab.items()}
        self.max_len = max([len(x) for x in self.vocab.keys()])
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

    def _load_dataset(self, path):
        dataset = []
        with open(path, "r") as input_data:
            for line in input_data:
                tokens = [x for x in json.loads(line)]
                token_ids = [self.vocab[x["surface"]] if x["surface"] \
                             in self.vocab else self.vocab["<UNK>"] for x in tokens]
                tokens_addition = [check_additon_word(x["raw"]) for x in tokens]
                targets = [self.tags[clear_target(x["target"])] for x in tokens]
                chars = [clear_char(token["raw"], self.max_len, self.char) for token in tokens]
                chars_addtion = [clear_char_addition(x, self.max_len) for x in tokens]
        dataset.append((token_ids, tokens_addition, chars, chars_addtion,  targets))
        return dataset

if __name__ == "__main__":
    DataProcessor("work/")
