import json
import tensorflow as tf
import os, sys
from collections import defaultdict

tf.flags.DEFINE_string("dest", "../work/", "destination dir")
tf.flags.DEFINE_string("clean_dir", "../work/", "data clean dir")
FLAGS = tf.flags.FLAGS

def main():
    token_freqs = defaultdict(int)
    list_file = os.listdir(FLAGS.clean_dir)
    for file in list_file:
        if not file.endswith(".clean"):
            continue
        with open(FLAGS.clean_dir + file) as fi:
            for line in fi:
                tokens = json.loads(line)
                for token in tokens:
                    token_freqs[token["surface"]] += 1
        

    singleton_txt = os.path.join(FLAGS.dest, 'singleton.txt')
    vocab_txt = os.path.join(FLAGS.dest, 'vocab.txt')

    singleton_ids = defaultdict(lambda: len(singleton_ids))
    vocab_ids = defaultdict(lambda: len(vocab_ids))
    vocab_ids["<UNK>"]
    for token, freq in token_freqs.items():
        if token.strip() == "":
            continue
        if freq == 1:
            singleton_ids[token] # keep the record of singletons
        vocab_ids[token] # all vocabulary goes here

    with open(singleton_txt, 'w') as singleton_fo:
        for token, _id in singleton_ids.items():
            singleton_fo.write("{}\t{}\n".format(token, _id))

    with open(vocab_txt, 'w') as vocab_fo:
        for token, _id in vocab_ids.items():
            vocab_fo.write("{}\t{}\n".format(token, _id))

    sys.stderr.write("Original Vocab Size:{}\nNumber of Singletons:{}\nNumber of Normal Vocabs:{}\n".format(
        len(token_freqs),
        len(singleton_ids),
        len(vocab_ids)
        ))

if __name__ == "__main__":
    main()
