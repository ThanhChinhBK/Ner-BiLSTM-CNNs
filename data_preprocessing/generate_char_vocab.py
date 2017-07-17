import os
import tensorflow as tf
import json
from collections import defaultdict

tf.flags.DEFINE_string("dest", "../work/", "destination dir")
tf.flags.DEFINE_string("clean_dir", "../work/", "data clean dir")
FLAGS = tf.flags.FLAGS

def main():
    vocab_txt = os.path.join(FLAGS.dest, 'vocab_char.txt')
    list_file = os.listdir(FLAGS.clean_dir)
    char_ids = defaultdict(lambda: len(char_ids))
    char_ids["<PAD>"]
    for file in list_file:
        if not file.endswith(".clean"):
            continue
        with open(FLAGS.clean_dir + file) as fi:
            for line in fi:
                tokens = json.loads(line)
                for token in tokens:
                    for char in token["surface"]:
                        char_ids[char]
    with open(vocab_txt, 'w') as vocab_fo:
        for vocab, _id in char_ids.items():
            vocab_fo.write("{}\t{}\n".format(vocab, _id))

if __name__ == "__main__":
    main()
